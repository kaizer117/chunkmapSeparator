# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 09:54:43 2025

@author: kavinduc
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from scipy.special import comb
from typing import Tuple, Optional
import time


class BezierCurveFitter:
    """Bézier curve fitter using Total Least Squares with Gauss-Newton optimization."""
    
    def __init__(self, degree: int):
        """
        Initialize the Bézier curve fitter.
        
        Parameters:
        -----------
        degree : int
            Degree of the Bézier curve to fit
        """
        self.degree = degree
        self.n_control_points = degree + 1
        self.control_points = None
        self.nodes = None
        
    def bernstein_matrix(self, t: np.ndarray, d: int) -> np.ndarray:
        """
        Create a Bernstein matrix of degree d for parameter values t.
        
        Parameters:
        -----------
        t : np.ndarray
            Parameter values (should be in [0, 1])
        d : int
            Degree of Bernstein polynomials
            
        Returns:
        --------
        np.ndarray
            Bernstein matrix B(t) of shape (len(t), d+1)
        """
        n = len(t)
        B = np.zeros((n, d + 1))
        
        # Avoid numerical issues with logarithms for large d
        if d < 23:
            # Direct computation using binomial coefficients
            for i in range(d + 1):
                binom = comb(d, i)
                B[:, i] = binom * (t ** i) * ((1 - t) ** (d - i))
        else:
            # Use logarithms for numerical stability
            for i in range(d + 1):
                log_binom = np.sum(np.log(np.arange(1, d + 1))) - \
                           np.sum(np.log(np.arange(1, i + 1))) - \
                           np.sum(np.log(np.arange(1, d - i + 1)))
                log_term = i * np.log(t) + (d - i) * np.log(1 - t)
                B[:, i] = np.exp(log_binom + log_term)
        
        return B
    
    def affine_invariant_angle_nodes(self, data: np.ndarray) -> np.ndarray:
        """
        Compute initial nodes using affine invariant angle method.
        
        Parameters:
        -----------
        data : np.ndarray
            Data points as (n, 2) array
            
        Returns:
        --------
        np.ndarray
            Initial node values in [0, 1]
        """
        n = len(data)
        
        # Compute covariance matrix and its inverse
        X_mean = np.mean(data, axis=0)
        X_centered = data - X_mean
        X_cov = (X_centered.T @ X_centered) / n
        A = np.linalg.inv(X_cov)
        
        # Compute metric distances between consecutive points
        V = data[1:, :] - data[:-1, :]
        t = np.sqrt(np.diag(V @ A @ V.T))
        
        # Compute metric distances for points two apart
        if n > 2:
            V2 = data[2:, :] - data[:-2, :]
            t2 = np.diag(V2 @ A @ V2.T)
        else:
            t2 = np.array([0])
        
        # Compute theta values (bending angles)
        theta = np.zeros(n - 1)
        
        for j in range(1, n - 1):
            # Handle potential numerical issues in acos
            cos_theta = (t[j-1]**2 + t[j]**2 - t2[j-1]) / (2 * t[j] * t[j-1])
            cos_theta = np.clip(cos_theta, -1, 1)  # Ensure within [-1, 1]
            theta[j] = min(np.pi - np.arccos(cos_theta), np.pi / 2)
        
        # Compute affine invariant spacing values h
        h = np.zeros(n - 1)
        
        if n > 1:
            h[0] = t[0] * (1 + (1.5 * theta[1] * t[1]) / (t[0] + t[1]))
            
            for j in range(1, n - 2):
                h[j] = t[j] * (1 + 
                              (1.5 * theta[j] * t[j-1]) / (t[j-1] + t[j]) +
                              (1.5 * theta[j+1] * t[j+1]) / (t[j] + t[j+1]))
            
            if n > 2:
                h[n-2] = t[n-2] * (1 + (1.5 * theta[n-2] * t[n-3]) / (t[n-3] + t[n-2]))
        
        # Normalize to [0, 1]
        h_cum = np.cumsum(np.insert(h, 0, 0))
        nodes = h_cum / h_cum[-1]
        
        return nodes
    
    def compute_control_points(self, data: np.ndarray, nodes: np.ndarray) -> np.ndarray:
        """
        Compute control points by solving linear least squares problem.
        
        Parameters:
        -----------
        data : np.ndarray
            Data points
        nodes : np.ndarray
            Parameter values for each data point
            
        Returns:
        --------
        np.ndarray
            Control points matrix
        """
        B = self.bernstein_matrix(nodes, self.degree)
        
        # Solve B * P = data for P (linear least squares)
        # Using numpy's lstsq for stability
        P, residuals, rank, s = lstsq(B, data)
        
        return P
    
    def compute_gradient(self, nodes: np.ndarray, control_points: np.ndarray) -> np.ndarray:
        """
        Compute gradient of Bézier curve at given nodes.
        
        Parameters:
        -----------
        nodes : np.ndarray
            Parameter values
        control_points : np.ndarray
            Control points matrix
            
        Returns:
        --------
        np.ndarray
            Gradient (derivative) matrix at each node
        """
        # Forward difference of control points
        delta_P = control_points[1:, :] - control_points[:-1, :]
        
        # Bernstein matrix for degree-1 curve
        B_lower = self.bernstein_matrix(nodes, self.degree - 1)
        
        # Gradient = degree * B_{d-1}(t) * ΔP
        gradient = self.degree * (B_lower @ delta_P)
        
        return gradient
    
    def gauss_newton_step(self, nodes: np.ndarray, control_points: np.ndarray, 
                         data: np.ndarray) -> np.ndarray:
        """
        Perform one Gauss-Newton step to update nodes.
        
        Parameters:
        -----------
        nodes : np.ndarray
            Current nodes
        control_points : np.ndarray
            Current control points
        data : np.ndarray
            Data points
            
        Returns:
        --------
        np.ndarray
            Updated nodes
        """
        # Compute current points on curve
        B = self.bernstein_matrix(nodes, self.degree)
        curve_points = B @ control_points
        
        # Compute residual
        residual = curve_points - data
        
        # Compute gradient (Jacobian)
        gradient = self.compute_gradient(nodes, control_points)
        
        # Compute Gauss-Newton update (simplified due to diagonal structure)
        # Δt_i = - (grad_i·residual_i) / (grad_i·grad_i)
        grad_dot_res = np.sum(gradient * residual, axis=1)
        grad_dot_grad = np.sum(gradient * gradient, axis=1)
        
        # Avoid division by zero
        grad_dot_grad = np.where(grad_dot_grad == 0, 1e-10, grad_dot_grad)
        
        delta_t = -grad_dot_res / grad_dot_grad
        new_nodes = nodes + delta_t
        
        # Ensure nodes stay in [0, 1] and maintain order
        new_nodes = np.clip(new_nodes, 0, 1)
        new_nodes = np.sort(new_nodes)
        
        return new_nodes
    
    def fit(self, data: np.ndarray, max_iter: int = 100, 
            tol: float = 1e-6, verbose: bool = False) -> dict:
        """
        Fit Bézier curve to data using Total Least Squares.
        
        Parameters:
        -----------
        data : np.ndarray
            Data points as (n, 2) array
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance
        verbose : bool
            Whether to print progress
        
        Functions contained:
        --------
        affine_invariant_angle_nodes
        compute_control_points
        bernstien_matrix
        gauss_newton_step
        
        Returns:
        --------
        dict
            Dictionary containing fitting results
        """
        n_points = len(data)
        
        if n_points <= self.degree:
            raise ValueError(f"Number of data points ({n_points}) must be greater than curve degree ({self.degree})")
        
        # Step 1: Initialize nodes using affine invariant angle method
        nodes = self.affine_invariant_angle_nodes(data)
        
        # Step 2: Initial control points
        control_points = self.compute_control_points(data, nodes)
        
        # Initial residual
        B = self.bernstein_matrix(nodes, self.degree)
        residual = B @ control_points - data
        prev_residual_norm = np.linalg.norm(residual, 'fro')
        
        if verbose:
            print(f"Iteration 0: Residual norm = {prev_residual_norm:.6e}")
        
        # Main optimization loop
        for iteration in range(1, max_iter + 1):
            # Step 3a: Update nodes using Gauss-Newton
            nodes = self.gauss_newton_step(nodes, control_points, data)
            
            # Step 3b: Update control points
            control_points = self.compute_control_points(data, nodes)
            
            # Compute new residual
            B = self.bernstein_matrix(nodes, self.degree)
            residual = B @ control_points - data
            residual_norm = np.linalg.norm(residual, 'fro')
            
            # Check convergence
            rel_change = abs(residual_norm - prev_residual_norm) / max(1.0, residual_norm)
            
            if verbose:
                print(f"Iteration {iteration}: Residual norm = {residual_norm:.6e}, "
                      f"Relative change = {rel_change:.6e}")
            
            if rel_change < tol:
                if verbose:
                    print(f"Converged after {iteration} iterations")
                break
            
            prev_residual_norm = residual_norm
        
        self.control_points = control_points
        self.nodes = nodes
        
        return {
            'control_points': control_points,
            'nodes': nodes,
            'residual_norm': residual_norm,
            'iterations': iteration
        }
    
    def evaluate_curve(self, t: np.ndarray = None, n_points: int = 100) -> np.ndarray:
        """
        Evaluate the fitted Bézier curve at specified parameter values.
        
        Parameters:
        -----------
        t : np.ndarray, optional
            Parameter values (if None, generates uniform values)
        n_points : int
            Number of points to generate if t is None
            
        Returns:
        --------
        np.ndarray
            Points on the curve
        """
        if self.control_points is None:
            raise ValueError("Must fit curve before evaluation")
        
        if t is None:
            t = np.linspace(0, 1, n_points)
        
        B = self.bernstein_matrix(t, self.degree)
        return B @ self.control_points
    
    def compute_approximation_error(self, data: np.ndarray) -> dict:
        """
        Compute approximation error metrics.
        
        Parameters:
        -----------
        data : np.ndarray
            Original data points
            
        Returns:
        --------
        dict
            Dictionary of error metrics
        """
        if self.control_points is None or self.nodes is None:
            raise ValueError("Must fit curve before computing error")
        
        # Points on curve at data nodes
        B = self.bernstein_matrix(self.nodes, self.degree)
        fitted_points = B @ self.control_points
        
        # Compute errors
        errors = fitted_points - data
        pointwise_distances = np.linalg.norm(errors, axis=1)
        
        return {
            'max_error': np.max(pointwise_distances),
            'mean_error': np.mean(pointwise_distances),
            'rmse': np.sqrt(np.mean(pointwise_distances ** 2)),
            'pointwise_errors': pointwise_distances
        }


def generate_example_data(n_points: int = 20, noise_std: float = 0.05, 
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


def plot_results(data: np.ndarray, fitter: BezierCurveFitter, 
                title: str = "Bézier Curve Fitting Results"):
    """
    Plot the fitting results.
    
    Parameters:
    -----------
    data : np.ndarray
        Original data points
    fitter : BezierCurveFitter
        Fitted curve object
    title : str
        Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Curve and control polygon
    ax = axes[0]
    
    # Generate smooth curve
    t_smooth = np.linspace(0, 1, 200)
    curve_smooth = fitter.evaluate_curve(t_smooth)
    
    # Plot data points
    ax.scatter(data[:, 0], data[:, 1], c='red', s=50, label='Data points', zorder=5)
    
    # Plot fitted curve
    ax.plot(curve_smooth[:, 0], curve_smooth[:, 1], 'b-', linewidth=2, label='Fitted curve')
    
    # Plot control points and polygon
    control_points = fitter.control_points
    ax.scatter(control_points[:, 0], control_points[:, 1], c='green', s=100, 
               marker='s', label='Control points', zorder=5)
    ax.plot(control_points[:, 0], control_points[:, 1], 'g--', alpha=0.5, 
            label='Control polygon')
    
    # Plot points on curve corresponding to data
    curve_at_nodes = fitter.evaluate_curve(fitter.nodes)
    ax.scatter(curve_at_nodes[:, 0], curve_at_nodes[:, 1], c='blue', s=30, 
               label='Curve at nodes', zorder=4)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Curve Fitting')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 2: Parameterization and errors
    ax = axes[1]
    
    # Compute errors
    errors = fitter.compute_approximation_error(data)
    
    # Plot parameter values
    ax.plot(fitter.nodes, np.zeros_like(fitter.nodes), 'ro', label='Nodes')
    ax.set_xlabel('Parameter t')
    ax.set_ylabel('Value')
    ax.set_title(f'Parameterization and Errors\nRMSE: {errors["rmse"]:.4f}')
    ax.grid(True, alpha=0.3)
    
    # Add error bars
    ax.errorbar(fitter.nodes, np.zeros_like(fitter.nodes), 
                yerr=errors['pointwise_errors'], fmt='none', 
                capsize=5, alpha=0.5, label='Error magnitude')
    
    ax.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    """Demonstrate the Bézier curve fitting algorithm."""
    print("=" * 60)
    print("Bézier Curve Fitting using Total Least Squares")
    print("=" * 60)
    
    # Generate example data
    print("\nGenerating example data (sigmoid curve with noise)...")
    data = generate_example_data(n_points=25, noise_std=0.03, curve_type='spiral')
    print(f"Data shape: {data.shape}")
    
    # Create and run fitter
    print("\nFitting cubic Bézier curve...")
    fitter = BezierCurveFitter(degree=5)
    
    start_time = time.time()
    result = fitter.fit(data, max_iter=50, tol=1e-6, verbose=True)
    elapsed_time = time.time() - start_time
    
    print(f"\nFitting completed in {elapsed_time:.3f} seconds")
    print(f"Final residual norm: {result['residual_norm']:.6e}")
    print(f"Number of iterations: {result['iterations']}")
    
    # Compute error metrics
    errors = fitter.compute_approximation_error(data)
    print("\nError metrics:")
    print(f"  Maximum error: {errors['max_error']:.6f}")
    print(f"  Mean error: {errors['mean_error']:.6f}")
    print(f"  RMSE: {errors['rmse']:.6f}")
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(data, fitter, title="Bézier Curve Fitting Example")
    
    print("\nControl points:")
    print(fitter.control_points)
    
    print("\nDone!")


if __name__ == "__main__":
    main()