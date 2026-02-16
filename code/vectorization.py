import numpy as np
import fileutils as f
from scipy.linalg import lstsq
from scipy.special import comb
from typing import List
import segmentation as seg
import copy

#for testing polygon segmentation
import colorspace as c

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

def get_xy_extent(data, padding=0, padding_type='absolute',output='height-width'):
    """
    Get (x_extent, y_extent) of 2D coordinate data with optional padding.
    
    Parameters:
    -----------
    data : array-like of shape (n_points, 2)
        Input coordinate data
    padding : float or tuple, default=0
        - float: same padding for both axes
        - tuple: (padding_x, padding_y)
    padding_type : str, default='absolute'
        'absolute', 'relative' (fraction of extent), or 'percentage'
    output : str, default='height-width'
        height-width: only gives out the weight and width px values
        viewbox: gives 4 values corresponding to the bounding box
        minmaxex: gives 6 values corresponding to minx, miny, maxx, maxy, extentx, extenty
    
    Returns:
    --------
    tuple : (x_0, y_0, x_extent, y_extent)
    """
    # Convert to numpy array
    data = np.asarray(data)
    
    # Handle 1D flattened array
    if data.ndim == 1:
        data = data.reshape(-1, 2)
    
    # Get raw extents
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    extents = max_vals - min_vals
    
    # Handle zero extents
    extents[extents == 0] = 1
    
    # Apply padding
    if padding != 0:
        if isinstance(padding, (int, float)):
            pad_x = pad_y = padding
        else:
            pad_x, pad_y = padding
        
        if padding_type == 'relative':
            pad_x *= extents[0]
            pad_y *= extents[1]
        elif padding_type == 'percentage':
            pad_x = (pad_x / 100) * extents[0]
            pad_y = (pad_y / 100) * extents[1]
        # please implement padding in a more meaningful
    if (output=="height-width"):
        return (max_vals[0], max_vals[1])
    elif (output=="viewbox"):# poorly implemented, pls fix, may even depricate at some point
        return (extents[0], extents[1],min_vals[0], min_vals[1])
    elif (output=="minmaxex"):
        return (min_vals[0],min_vals[1],max_vals[0], max_vals[1],extents[0], extents[1])

def controlPointFormatter(ctrlPoints):
    """
    another variation of reshapeCtrlSVG, this one will, calculate end/start points for subsequent
    curves as well as reshaping the ctrlPoint array such that the array will have the following
    format:

    [4 points],[3 points],[3 points]

    Parameters:
    --------
    ctrlPoints : nd.array or list
    
    Returns:
    --------
    list: As specified above
    """
    # deepcopy the ctrl points
    ctrlPointCopy = copy.deepcopy(ctrlPoints)
    # create and calculate the mid-points
    def calcMidPoint (p1,p2):
        return np.array([(p1[0]+p2[0])/2,(p1[1]+p2[1])/2])
    try:
        endPoints = list(map(lambda i: calcMidPoint(ctrlPointCopy[i][3],ctrlPointCopy[i+1][0]),range(len(ctrlPointCopy)-1)))
    except IndexError as e:
        raise e
    
    # assign calculated midpoints to the end of every ctrpoint array
    for i in range(len(ctrlPointCopy)-1):
        ctrlPointCopy[i][3]=endPoints[i]
    
    for i in range(1,len(ctrlPointCopy)):
        ctrlPointCopy[i]=ctrlPointCopy[i][1:]

    return ctrlPointCopy

def visualizeSegmentation(data,stroke_width = 1):
    """
    Take the segmented array and output a svg file visualising the segments. Purely for
    testing purposes.
    
    Parameters:
    -----------
    segments : array of arrays containing the segmented data
    
    Returns:
    --------
    None
    """
    svgHandler = f.SVGHandler()

    shapeTuple = get_xy_extent(data,output="minmaxex")

    # shift the shape to the origin
    data[:,0]=data[:,0]-shapeTuple[0]
    data[:,1]=data[:,1]-shapeTuple[1]

    svgHandler.initialize(img_size=(shapeTuple[4],shapeTuple[5]))

    segments = seg.polygon_curvature_segmentation(data,min_chunk=5,max_chunk=15,smooth_sigma=2)

    #actual vecotirzation
    ctrlCon = list(map(fitBezierCurve,segments))

    #prepare the data set and ctrl points for svg output
    def link_arrays(arrays):
        # inefficient as heck
        len_arrays = len(arrays)
        return np.array(list(map(lambda i:np.concatenate((arrays[i%len_arrays],arrays[(i+1)%len_arrays][0].reshape(1,2))),range(len_arrays))),dtype=object)
    
    ctrlCon = controlPointFormatter(ctrlCon)
    segments = link_arrays(segments)

    colors = c.getRandColours(len(segments))
    baseStyle = {
                        "stroke":None,
                        "stroke-width": str(stroke_width),
                        "fill": "none"
                        }
    
    
    
    styleGuideDict = {"bezierStyle":{
                        "stroke":"#e08776",
                        "stroke-width": str(stroke_width*0.8),
                        "stroke-dasharray":"3 2",
                        "stroke-opacity":"0.7",
                        "fill": "none"
                        }}

    for i in range (len(segments)):
        baseStyle["stroke"]=colors[i]
        styleGuideDict['thin-line-'+str(i)]=baseStyle.copy()
        svgHandler.drawline(segments[i],'thin-line-'+str(i))
        #svgHandler.drawCubicBezierSingular(ctrlCon[i]," bezierStyle")
    svgHandler.drawcubicbezier(ctrlCon," bezierStyle")
    svgHandler.addstyletag(styleGuideDict)
    svgHandler.compose()
    svgHandler.save()

    return None

if (__name__=="__main__"):
    data = generateDatapoints(100,curve_type="circle")*100+5
    segments = polygonSegmentation(data,9)
    print('Hi')