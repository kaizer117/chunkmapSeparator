import numpy as np
import fileutils as f
from scipy.linalg import lstsq
from scipy.special import comb
from typing import List

def generateDatapoints(n_points: int = 20, noise_std: float = 0.05, 
                         curve_type: str = 'sigmoid') -> np.ndarray:
    """
    Generate example 2D data for testing.
    
    Parameters:
    -----------
    n_points : int
        Number of data points
    noise_std : float
        Standard deviation of Gaussian noise
    curve_type : str
        Type of curve: 'sigmoid', 'circle', or 'spiral'
        
    Returns:
    --------
    np.ndarray
        Generated data points
    """
    np.random.seed()
    
    if curve_type == 'sigmoid':
        t = np.linspace(-2, 2, n_points)
        x = t
        y = 1 / (1 + np.exp(-t)) + np.random.randn(n_points) * noise_std
    
    elif curve_type == 'circle':
        theta = np.linspace(0, 2 * np.pi, n_points)
        x = np.cos(theta) + np.random.randn(n_points) * noise_std
        y = np.sin(theta) + np.random.randn(n_points) * noise_std
    
    elif curve_type == 'spiral':
        theta = np.linspace(0, 3 * np.pi, n_points)
        r = 0.5 * theta / (3 * np.pi)
        x = r * np.cos(theta) + np.random.randn(n_points) * noise_std
        y = r * np.sin(theta) + np.random.randn(n_points) * noise_std
    
    else:
        raise ValueError(f"Unknown curve type: {curve_type}")
    
    # Stack and scale to reasonable range
    data = np.column_stack([x, y])
    data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    
    return data
    
    
def sampleT(datapoints,method='square_root_distance'):

    array=[]
    if (method=='square_root_distance'):
        for i in range(len(datapoints)-1):
            array.append(np.sqrt((datapoints[i][0]-datapoints[i+1][0])**2+(datapoints[i][1]-datapoints[i+1][1])**2))
    elif (method=='squared_distance'):
        for i in range(len(datapoints)-1):
            array.append((datapoints[i][0]-datapoints[i+1][0])**2+(datapoints[i][1]-datapoints[i+1][1])**2)
    elif (method=='unsquared_sum'):
        for i in range(len(datapoints)-1):
            array.append(np.abs(datapoints[i][0]-datapoints[i+1][0])+np.abs(datapoints[i][1]-datapoints[i+1][1]))
    
    return np.insert(np.cumsum(array),0,0)/np.sum(array)



def computCurvature(data):
    """
    This function calculates the curvature of a plane curve.
    Source: https://en.wikipedia.org/wiki/Curvature#Plane_curves

    This is not completely accurate
    
    :param data: the coordinate data
    """

    # extract x and y data
    x = data[:, 0]
    y = data[:, 1]

    # calculate gradients
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # calculate curvature and change in curvature
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    #change_in_curvature = np.diff(curvature)

    stats = {
        'n': len(curvature),
        'mean': np.mean(curvature),
        'median': np.median(curvature),
        'std': np.std(curvature),
        'min': min(curvature),
        'max': max(curvature)
    }

    return curvature, stats

def getPreciseCrossings(values: np.ndarray, 
                         threshold: float) -> List[float]:
    """
    Get more precise crossing positions using linear interpolation.
    
    Returns:
    --------
    List of fractional indices 
    """
    values = np.asarray(values)
    
    
    precise_crossings = []
    
    for i in range(1, len(values)):
        v_prev = values[i-1]
        v_curr = values[i]
        
        # Check for rising crossing
        if v_prev < threshold <= v_curr:
            # Linear interpolation
            precise_idx = (i-1)
            precise_crossings.append(precise_idx)
        
        # Check for falling crossing
    
        # if v_prev > threshold >= v_curr:
        #     # Linear interpolation
        #     precise_idx = (i-1)
        #     precise_crossings.append(precise_idx)

    return precise_crossings

def polygonSegmentation(data, chunk_size=10):
    """
    Blind segmentation
    Curvature aware segmentation
    """

    # 1. Implemeting blind segmentation
    n_points = len(data)
    while (n_points <= 2*chunk_size):
        chunk_size = int(chunk_size/1.5)
    if (chunk_size <= 4):
        print(f'{chunk_size} is too small to segment the polygon of length {n_points}\n')
        return [data]
    
    # readjusting the chunk size
    chunks = n_points//chunk_size+1
    chunk_size = n_points // chunks

    # populating indexes and pushing the remainders to the final chunk
    indexes = list(map(lambda i:i*chunk_size,range(chunks)))
    indexes.append(n_points-1)

    # using map object to splice the data array

    return list(map(lambda i: data[indexes[i]:indexes[i+1]],range(len(indexes)-1)))

def fitBezierCurve(data: np.ndarray, use_cache: bool = True) -> np.ndarray:
    """
    Fit a Bézier curve to 2D data using Total Least Squares with Gauss-Newton optimization.
    This algorithm is a direct MATLAB-to-Python translation of Tim. A, Pastva's algorithm.
    Source: https://calhoun.nps.edu/entities/publication/8126011c-a7ec-4cad-8372-4c971bf915a9
    
    Certain cacheing and early termination techniques are used to improve the original speed
    by 20-30%. Further imporovments can be made alongside more optimizations for cubic
    bezier approximation.

    This algorithm provides higher order bezier curves whereas SVG files only support up to 
    cubic bezeier curves (degree = 3)
    
    Parameters:
    -----------
    data : np.ndarray
        (n, 2) array of 2D points
    use_cache : bool
        Whether to cache Bernstein matrices for speed
    
    Returns:
    --------
    np.ndarray
        (degree+1, 2) control points of the fitted Bézier curve
    """
    
    # ============================================================================
    # Helper functions
    # ============================================================================
    
    def bernstein_matrix(t: np.ndarray, d: int) -> np.ndarray:
        """Bernstein matrix of degree d for parameter values t."""
        n = len(t)
        B = np.zeros((n, d + 1))
        
        if d < 23:
            for i in range(d + 1):
                binom = comb(d, i)
                B[:, i] = binom * (t ** i) * ((1 - t) ** (d - i))
        else:
            for i in range(d + 1):
                log_binom = np.sum(np.log(np.arange(1, d + 1))) - \
                           np.sum(np.log(np.arange(1, i + 1))) - \
                           np.sum(np.log(np.arange(1, d - i + 1)))
                log_term = i * np.log(t) + (d - i) * np.log(1 - t)
                B[:, i] = np.exp(log_binom + log_term)
        
        return B
    
    def compute_control_points(data: np.ndarray, nodes: np.ndarray, d: int, 
                               cache: dict = None) -> np.ndarray:
        """Compute control points by solving B*P = data."""
        if use_cache and cache is not None:
            cache_key = (tuple(nodes.round(6)), d)
            if cache_key in cache:
                B = cache[cache_key].copy()
            else:
                B = bernstein_matrix(nodes, d)
                if len(cache) < 100:
                    cache[cache_key] = B.copy()
        else:
            B = bernstein_matrix(nodes, d)
        
        P, _, _, _ = lstsq(B, data)
        return P
    
    def compute_gradient(nodes: np.ndarray, control_points: np.ndarray, d: int) -> np.ndarray:
        """Compute gradient of Bézier curve at given nodes."""
        delta_P = control_points[1:, :] - control_points[:-1, :]
        B_lower = bernstein_matrix(nodes, d - 1)
        return d * (B_lower @ delta_P)
    
    def gauss_newton_step(nodes: np.ndarray, control_points: np.ndarray, 
                         data: np.ndarray, d: int) -> np.ndarray:
        """Perform one Gauss-Newton step to update nodes."""
        B = bernstein_matrix(nodes, d)
        curve_points = B @ control_points
        residual = curve_points - data
        gradient = compute_gradient(nodes, control_points, d)
        
        grad_dot_res = np.sum(gradient * residual, axis=1)
        grad_dot_grad = np.sum(gradient * gradient, axis=1)
        grad_dot_grad = np.where(grad_dot_grad == 0, 1e-10, grad_dot_grad)
        
        delta_t = -grad_dot_res / grad_dot_grad
        max_step = 0.1
        step_norm = np.sqrt(np.mean(delta_t**2))
        if step_norm > max_step:
            delta_t = delta_t * (max_step / step_norm)
        
        new_nodes = nodes + delta_t
        new_nodes = np.clip(new_nodes, 0, 1)
        return np.sort(new_nodes)
    
    def affine_invariant_nodes(data: np.ndarray) -> np.ndarray:
        """Compute initial nodes using affine invariant angle method."""
        n = len(data)
        X_mean = np.mean(data, axis=0)
        X_centered = data - X_mean
        X_cov = (X_centered.T @ X_centered) / n
        A = np.linalg.inv(X_cov)
        
        V = data[1:, :] - data[:-1, :]
        t = np.sqrt(np.diag(V @ A @ V.T))
        
        if n > 2:
            V2 = data[2:, :] - data[:-2, :]
            t2 = np.diag(V2 @ A @ V2.T)
        else:
            t2 = np.array([0])
        
        theta = np.zeros(n - 1)
        for j in range(1, n - 1):
            cos_theta = (t[j-1]**2 + t[j]**2 - t2[j-1]) / (2 * t[j] * t[j-1])
            cos_theta = np.clip(cos_theta, -1, 1)
            theta[j] = min(np.pi - np.arccos(cos_theta), np.pi / 2)
        
        h = np.zeros(n - 1)
        if n > 1:
            h[0] = t[0] * (1 + (1.5 * theta[1] * t[1]) / (t[0] + t[1]))
            for j in range(1, n - 2):
                h[j] = t[j] * (1 + (1.5 * theta[j] * t[j-1]) / (t[j-1] + t[j]) +
                               (1.5 * theta[j+1] * t[j+1]) / (t[j] + t[j+1]))
            if n > 2:
                h[n-2] = t[n-2] * (1 + (1.5 * theta[n-2] * t[n-3]) / (t[n-3] + t[n-2]))
        
        h_cum = np.cumsum(np.insert(h, 0, 0)) # Does this have any meaningful impact ?
        return h_cum / h_cum[-1]
    
    def select_degree_fixed(n_points: int) -> int:
        """
        Select degree based on fixed heuristic.
        """
        min_degree = n_points//10
        max_degree = n_points//5
        
        # Heuristic 1: Based on square root of points
        degree_sqrt = int(np.round(np.sqrt(n_points)))
        
        # Heuristic 2: Based on points per segment
        degree_segment = int(np.round(n_points / 10))
        
        # Take average and clamp to bounds
        degree = int(np.round((degree_sqrt + degree_segment) / 2))
        degree = max(min_degree, min(max_degree, degree))
        
        # Ensure degree < number of points
        degree = min(degree, n_points - 1)
        
        return degree
    
    # ============================================================================
    # Main fitting routine
    # ============================================================================
    
    n_points = len(data)
    
    # # Determine degree if not provided
    # # if degree is None:
    # #     degree = min(7, max(2, n_points // 4))
    # # degree = min(degree, n_points - 1)

    # # define min and max degrees based on the maximum percision point to degree scale
    # min_degree = n_points//10
    # max_degree = n_points//5
    
    # degree =select_degree_fixed(n_points)

    degree = 3 # essentially clamp the function to do cubic bezier approximation for now.
    
    # Initialize nodes
    nodes = affine_invariant_nodes(data)
    
    # Initialize cache if needed
    cache = {} if use_cache else None
    
    # Initial control points
    control_points = compute_control_points(data, nodes, degree, cache)
    
    # Optimization loop
    max_iter = 50
    tol = 1e-6
    prev_residual_norm = float('inf')
    
    for iteration in range(max_iter):
        # Update nodes
        nodes = gauss_newton_step(nodes, control_points, data, degree)
        
        # Update control points
        control_points = compute_control_points(data, nodes, degree, cache)
        
        # Check convergence
        B = bernstein_matrix(nodes, degree)
        residual_norm = np.linalg.norm(B @ control_points - data, 'fro')
        
        if iteration > 0:
            rel_change = abs(residual_norm - prev_residual_norm) / max(1.0, residual_norm)
            if rel_change < tol:
                break
        
        prev_residual_norm = residual_norm
    
    return control_points

def vectorizeContour(data):
    """
    Creates a list of lists for cubic bezeir control points for each segment of
    one contour
    
    :param data: raster data, 2d coordinates of the contour
    """
    # Segment polygon data
    segmentedData = polygonSegmentation(data)

    # Use map function to fit cubic bezier curve
    ctrlCon = list(map(fitBezierCurve,segmentedData))

    return np.array(ctrlCon)

def reshapeCtrlSVG(ctrlCon):
    """
    SVG cubic bezier curves accept control points in a very weird way. This function acts
    as a bridge between the output of vectorizeContour and other svg methods

    Most probably will be deprciated when the fitbeziercruves get revamed
    """
    return np.concat([ctrlCon[0],np.concat(ctrlCon[1:,1:])])

def vectorizeContours():
    pass

if (__name__=="__main__"):
    
    x = np.random.random(int(np.floor(np.random.random()*100)+30))
    polygonSegmentation(x)
    
    
    
    
    
    
    
    print('Hi')