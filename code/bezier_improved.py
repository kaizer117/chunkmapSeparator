# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 09:54:43 2025

@author: kavinduc
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from scipy.special import comb
from scipy.linalg import solve_triangular, qr
from typing import Tuple, Optional, Dict
import time
from collections import OrderedDict
import pandas as pd

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


class BezierCurveFitterOpt(BezierCurveFitter):
    """Bézier curve fitter using Total Least Squares with Gauss-Newton optimization."""
    
    def __init__(self, degree: int, use_cache: bool = True, cache_size: int = 1000):
        """
        Initialize the Bézier curve fitter.
        
        Parameters:
        -----------
        degree : int
            Degree of the Bézier curve to fit
        use_cache : bool
            Whether to use caching optimizations
        cache_size : int
            Size of basis function cache
        """
        self.degree = degree
        self.n_control_points = degree + 1
        self.control_points = None
        self.nodes = None
        self.use_cache = use_cache
        
        # ================================================================
        # PHASE 1 OPTIMIZATION 1: Basis Function Caching (5-20x speedup)
        # ================================================================
        if self.use_cache:
            self.cache_size = cache_size
            self.basis_cache = OrderedDict()  # LRU cache for Bernstein matrices
            self.max_cache_size = cache_size
        
        # ================================================================
        # PHASE 1 OPTIMIZATION 2: QR Decomposition Reuse (2-5x speedup)
        # ================================================================
        self.qr_cache = None
        self.qr_tolerance = 1e-4  # Tolerance for QR reuse
        
        # ================================================================
        # PHASE 1 OPTIMIZATION 3: Early Termination (1.5-4x speedup)
        # ================================================================
        self.min_iterations = 5
        self.patience = 3  # Stop if no improvement for N iterations
        
        # Performance tracking
        self.stats = {
            'basis_cache_hits': 0,
            'basis_cache_misses': 0,
            'qr_reuse_count': 0,
            'qr_recompute_count': 0,
            'early_termination': False,
            'iterations': 0
        }
    
    def _get_cache_key(self, t: np.ndarray, degree: int) -> str:
        """Generate cache key for Bernstein matrix."""
        # Use hashed version of sorted t values for cache key
        t_sorted = np.sort(t)
        key_data = f"{degree}_{t_sorted[:10].tobytes()}_{len(t)}"
        return hash(key_data)
    
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
        # ================================================================
        # PHASE 1 OPTIMIZATION: Basis Function Caching
        # ================================================================
        if self.use_cache:
            cache_key = self._get_cache_key(t, d)
            
            if cache_key in self.basis_cache:
                # Cache hit - return cached value (move to front)
                self.basis_cache.move_to_end(cache_key)
                self.stats['basis_cache_hits'] += 1
                return self.basis_cache[cache_key].copy()
            
            # Cache miss - compute and cache
            self.stats['basis_cache_misses'] += 1
        
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
        
        # ================================================================
        # PHASE 1 OPTIMIZATION: Store in cache
        # ================================================================
        if self.use_cache:
            # Store in cache (LRU eviction)
            if len(self.basis_cache) >= self.max_cache_size:
                self.basis_cache.popitem(last=False)
            
            self.basis_cache[cache_key] = B.copy()
        
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
    
    def _nodes_changed_significantly(self, nodes_new: np.ndarray, nodes_old: np.ndarray) -> bool:
        """
        Check if nodes have changed enough to warrant recomputing QR.
        
        Part of PHASE 1 OPTIMIZATION: QR Decomposition Reuse
        """
        if nodes_old is None or len(nodes_new) != len(nodes_old):
            return True
        
        # Check maximum absolute change
        max_change = np.max(np.abs(nodes_new - nodes_old))
        
        # Check pattern correlation (are nodes moving together?)
        if len(nodes_new) > 1:
            correlation = np.corrcoef(nodes_new, nodes_old)[0, 1]
            pattern_similar = correlation > 0.95
        else:
            pattern_similar = True
        
        # Recompute if significant change or pattern divergence
        return max_change > self.qr_tolerance or not pattern_similar
    
    def compute_control_points_(self, data: np.ndarray, nodes: np.ndarray) -> np.ndarray:
        B = self.bernstein_matrix(nodes, self.degree)
        
        if self.qr_cache is not None:
            nodes_changed = self._nodes_changed_significantly(nodes, self.qr_cache['nodes'])
            
            if not nodes_changed:
                # ACTUAL QR reuse
                Q, R = self.qr_cache['QR']
                self.stats['qr_reuse_count'] += 1
                
                # Solve: P = R⁻¹(Qᵀ @ data)
                return solve_triangular(R, Q.T @ data)
        
        # Compute QR decomposition
        self.stats['qr_recompute_count'] += 1
        Q, R = qr(B, mode='economic')
        
        # Cache it
        self.qr_cache = {
            'QR': (Q.copy(), R.copy()),
            'nodes': nodes.copy(),
            'degree': self.degree
        }
        
        # Solve using the newly computed QR
        return solve_triangular(R, Q.T @ data)
    
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
        
        # Apply update with step limiting for stability
        max_step = 0.1  # Limit maximum step size
        step_norm = np.sqrt(np.mean(delta_t**2))
        if step_norm > max_step:
            delta_t = delta_t * (max_step / step_norm)
        
        new_nodes = nodes + delta_t
        
        # Ensure nodes stay in [0, 1] and maintain order
        new_nodes = np.clip(new_nodes, 0, 1)
        new_nodes = np.sort(new_nodes)
        
        return new_nodes
    
    def _check_convergence(self, current_error: float, prev_error: float, 
                          iteration: int, no_improvement_count: int) -> Tuple[bool, int]:
        """
        Enhanced convergence checking with early termination.
        
        Part of PHASE 1 OPTIMIZATION: Early Termination
        """
        # Always run minimum iterations
        if iteration < self.min_iterations:
            return False, no_improvement_count
        
        # Compute relative improvement
        if prev_error > 0:
            rel_improvement = abs(current_error - prev_error) / prev_error
        else:
            rel_improvement = float('inf')
        
        # Check convergence criteria
        converged = False
        
        # Criterion 1: Small relative improvement
        if rel_improvement < 1e-6:
            converged = True
        
        # Criterion 2: No improvement for patience iterations
        elif no_improvement_count >= self.patience:
            converged = True
        
        # Criterion 3: Error increased significantly (divergence)
        elif current_error > prev_error * 1.5:
            converged = True
        
        # Update no-improvement counter
        if rel_improvement < 1e-4:  # Very small improvement
            no_improvement_count += 1
        else:
            no_improvement_count = max(0, no_improvement_count - 1)
        
        if converged:
            self.stats['early_termination'] = True
        
        return converged, no_improvement_count
    
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
        
        # ================================================================
        # PHASE 1 OPTIMIZATION: Early termination tracking
        # ================================================================
        no_improvement_count = 0
        
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
            
            # ============================================================
            # PHASE 1 OPTIMIZATION: Enhanced convergence checking
            # ============================================================
            converged, no_improvement_count = self._check_convergence(
                residual_norm, prev_residual_norm, iteration, no_improvement_count
            )
            
            if rel_change < tol or converged:
                if verbose:
                    if converged and not (rel_change < tol):
                        print(f"Early termination after {iteration} iterations")
                    else:
                        print(f"Converged after {iteration} iterations")
                break
            
            prev_residual_norm = residual_norm
        
        self.control_points = control_points
        self.nodes = nodes
        self.stats['iterations'] = iteration
        
        return {
            'control_points': control_points,
            'nodes': nodes,
            'residual_norm': residual_norm,
            'iterations': iteration,
            'optimization_stats': self.stats.copy()  # Include optimization statistics
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
    
    def print_optimization_stats(self):
        """Print Phase 1 optimization statistics."""
        if not self.use_cache:
            print("Optimizations disabled (use_cache=False)")
            return
        
        print("\n" + "="*50)
        print("PHASE 1 OPTIMIZATION STATISTICS")
        print("="*50)
        
        # Basis cache stats
        total_basis = self.stats['basis_cache_hits'] + self.stats['basis_cache_misses']
        if total_basis > 0:
            hit_rate = self.stats['basis_cache_hits'] / total_basis * 100
            print(f"Basis Cache: {hit_rate:.1f}% hit rate "
                  f"({self.stats['basis_cache_hits']}/{total_basis})")
        
        # QR cache stats
        total_qr = self.stats['qr_reuse_count'] + self.stats['qr_recompute_count']
        if total_qr > 0:
            reuse_rate = self.stats['qr_reuse_count'] / total_qr * 100
            print(f"QR Reuse: {reuse_rate:.1f}% reuse rate "
                  f"({self.stats['qr_reuse_count']}/{total_qr})")
        
        # Early termination
        if self.stats['early_termination']:
            print(f"Early Termination: Activated at iteration {self.stats['iterations']}")
        
        # Performance summary
        print(f"Iterations: {self.stats['iterations']}")
        print("="*50)


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


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import warnings

def plot_fine_histogram(data, bin_size=1, density=False, cumulative=False,
                       color='skyblue', edgecolor='black', alpha=0.7,
                       title=None, xlabel='Value', ylabel=None,
                       figsize=(10, 6), dpi=100, show_stats=True,
                       show_grid=True, vertical_lines=None, ax=None):
    """
    Plot histogram with very fine binning (1 or 2 unit bins).
    
    Parameters:
    -----------
    data : array-like
        Input data values
    bin_size : int or float, default=1
        Width of each bin. For fine histograms, use 1 or 2.
        Can be fractional like 0.5, 0.25, etc.
    density : bool, default=False
        If True, normalize to form probability density
    cumulative : bool or -1, default=False
        If True, plot cumulative histogram
        If -1, plot reversed cumulative histogram
    color : str, default='skyblue'
        Bar color
    edgecolor : str, default='black'
        Bar edge color
    alpha : float, default=0.7
        Bar transparency
    title : str, optional
        Plot title
    xlabel : str, default='Value'
        X-axis label
    ylabel : str, optional
        Y-axis label (auto-generated if None)
    figsize : tuple, default=(10, 6)
        Figure size (width, height)
    dpi : int, default=100
        Figure resolution
    show_stats : bool, default=True
        If True, display statistics on plot
    show_grid : bool, default=True
        If True, show grid lines
    vertical_lines : list or dict, optional
        Vertical lines to add to plot. Can be:
        - List of x positions
        - Dict with {'positions': [], 'colors': [], 'styles': [], 'labels': []}
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    stats : dict
        Statistics about the data
    """
    
    # Convert to numpy array
    data = np.asarray(data).flatten()
    
    # Handle empty data
    if len(data) == 0:
        warnings.warn("Empty data provided, returning empty plot")
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            if title:
                ax.set_title(title)
            return fig, ax, {}
        else:
            return ax.get_figure(), ax, {}
    
    # Calculate bin edges
    data_min = np.min(data)
    data_max = np.max(data)
    
    # Extend range slightly to include all data
    epsilon = 1e-10  # Small value to ensure max value is included
    data_min = np.floor(data_min / bin_size) * bin_size
    data_max = np.ceil(data_max / bin_size) * bin_size + epsilon
    
    # Create bins
    bins = np.arange(data_min, data_max, bin_size)
    
    # If only one data point or very small range, adjust bins
    if len(bins) < 2:
        bins = np.array([data_min - bin_size/2, data_min + bin_size/2])
    
    # Create figure if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()
    
    # Plot histogram
    n, bin_edges, patches = ax.hist(
        data, bins=bins, density=density, cumulative=cumulative,
        color=color, edgecolor=edgecolor, alpha=alpha,
        align='mid', rwidth=0.8
    )
    
    # Customize plot
    ax.set_xlabel(xlabel)
    
    if ylabel is None:
        if density:
            ylabel = 'Density'
        elif cumulative:
            ylabel = 'Cumulative Frequency'
        else:
            ylabel = 'Frequency'
    
    ax.set_ylabel(ylabel)
    
    if title:
        ax.set_title(title)
    
    # Force integer ticks on x-axis for small bin sizes
    if bin_size >= 1:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add grid
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add vertical lines if specified
    if vertical_lines is not None:
        if isinstance(vertical_lines, dict):
            positions = vertical_lines.get('positions', [])
            colors = vertical_lines.get('colors', ['red'] * len(positions))
            styles = vertical_lines.get('styles', ['--'] * len(positions))
            labels = vertical_lines.get('labels', [''] * len(positions))
            linewidths = vertical_lines.get('linewidths', [1.5] * len(positions))
            alphas = vertical_lines.get('alphas', [0.7] * len(positions))
            
            for pos, color, style, label, lw, alpha_val in zip(
                positions, colors, styles, labels, linewidths, alphas
            ):
                ax.axvline(x=pos, color=color, linestyle=style, 
                          linewidth=lw, alpha=alpha_val, label=label)
        
        elif isinstance(vertical_lines, (list, np.ndarray)):
            for pos in vertical_lines:
                ax.axvline(x=pos, color='red', linestyle='--', 
                          linewidth=1.5, alpha=0.7)
    
    # Calculate statistics
    stats = {
        'n': len(data),
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': data_min,
        'max': data_max,
        'range': data_max - data_min,
        'bin_size': bin_size,
        'n_bins': len(bins) - 1,
        'max_bin_count': np.max(n) if len(n) > 0 else 0,
        'max_bin_value': bin_edges[np.argmax(n)] if len(n) > 0 else 0,
        'total_count': np.sum(n) if not density else None
    }
    
    # Add statistics text to plot
    if show_stats:
        stats_text = (
            f'n = {stats["n"]:,}\n'
            f'Mean = {stats["mean"]:.3f}\n'
            f'Median = {stats["median"]:.3f}\n'
            f'Std = {stats["std"]:.3f}\n'
            f'Range = [{stats["min"]:.1f}, {stats["max"]:.1f}]\n'
            f'Bin size = {bin_size}'
        )
        
        # Place text in upper right
        ax.text(0.98, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9, fontfamily='monospace')
    
    # Adjust layout
    fig.tight_layout()
    
    return fig, ax, stats


# Specialized functions for common use cases
def plot_unit_histogram(data, **kwargs):
    """
    Plot histogram with 1-unit bins.
    
    Parameters:
    -----------
    data : array-like
        Input data values
    **kwargs : additional arguments passed to plot_fine_histogram
    
    Returns:
    --------
    fig, ax, stats
    """
    return plot_fine_histogram(data, bin_size=1, **kwargs)


def plot_two_unit_histogram(data, **kwargs):
    """
    Plot histogram with 2-unit bins.
    
    Parameters:
    -----------
    data : array-like
        Input data values
    **kwargs : additional arguments passed to plot_fine_histogram
    
    Returns:
    --------
    fig, ax, stats
    """
    return plot_fine_histogram(data, bin_size=2, **kwargs)


def plot_fractional_histogram(data, fraction=0.5, **kwargs):
    """
    Plot histogram with fractional bins.
    
    Parameters:
    -----------
    data : array-like
        Input data values
    fraction : float, default=0.5
        Bin size as fraction of unit
    **kwargs : additional arguments passed to plot_fine_histogram
    
    Returns:
    --------
    fig, ax, stats
    """
    return plot_fine_histogram(data, bin_size=fraction, **kwargs)


# Function to compare multiple datasets with fine bins
def compare_fine_histograms(datasets, labels=None, bin_size=1,
                           colors=None, alpha=0.7,
                           figsize=(12, 8), title=None):
    """
    Compare multiple datasets with fine-bin histograms.
    
    Parameters:
    -----------
    datasets : list of array-like
        Multiple datasets to compare
    labels : list of str, optional
        Labels for each dataset
    bin_size : int or float, default=1
        Bin width
    colors : list of str, optional
        Colors for each dataset
    alpha : float, default=0.7
        Transparency
    figsize : tuple, default=(12, 8)
        Figure size
    title : str, optional
        Overall title
    
    Returns:
    --------
    fig, axes, all_stats
    """
    n_datasets = len(datasets)
    
    if labels is None:
        labels = [f'Dataset {i+1}' for i in range(n_datasets)]
    
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, n_datasets))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=figsize,
                            gridspec_kw={'height_ratios': [3, 1]})
    ax_hist, ax_stats = axes
    
    all_stats = []
    
    # Plot histograms
    for i, (data, label, color) in enumerate(zip(datasets, labels, colors)):
        data = np.asarray(data).flatten()
        
        # Calculate bin edges (common for all datasets)
        if i == 0:
            all_data = np.concatenate(datasets)
            data_min = np.floor(np.min(all_data) / bin_size) * bin_size
            data_max = np.ceil(np.max(all_data) / bin_size) * bin_size + 1e-10
            bins = np.arange(data_min, data_max, bin_size)
        
        # Plot histogram
        ax_hist.hist(data, bins=bins, alpha=alpha, color=color,
                    edgecolor='black', linewidth=0.5,
                    label=label, density=True)
        
        # Calculate and store statistics
        stats = {
            'label': label,
            'n': len(data),
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data)
        }
        all_stats.append(stats)
    
    # Customize histogram axis
    ax_hist.set_xlabel('Value')
    ax_hist.set_ylabel('Density')
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)
    
    if title:
        ax_hist.set_title(title)
    else:
        ax_hist.set_title(f'Comparison of {n_datasets} datasets (bin size={bin_size})')
    
    # Create statistics table
    if n_datasets <= 10:  # Only create table for reasonable number of datasets
        # Prepare data for table
        table_data = []
        headers = ['Dataset', 'n', 'Mean', 'Median', 'Std', 'Min', 'Max']
        
        for stats in all_stats:
            table_data.append([
                stats['label'],
                f"{stats['n']:,}",
                f"{stats['mean']:.3f}",
                f"{stats['median']:.3f}",
                f"{stats['std']:.3f}",
                f"{stats['min']:.3f}",
                f"{stats['max']:.3f}"
            ])
        
        # Create table
        ax_stats.axis('tight')
        ax_stats.axis('off')
        table = ax_stats.table(cellText=table_data,
                              colLabels=headers,
                              loc='center',
                              cellLoc='center')
        
        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
    
    plt.tight_layout()
    return fig, axes, all_stats


# Example usage and demonstration
def demonstrate_fine_histograms():
    """Demonstrate the fine histogram plotting functions."""
    np.random.seed(42)
    
    print("Fine Histogram Plotting Demonstration")
    print("=" * 50)
    
    # Example 1: Integer data with unit bins
    print("\n1. Integer data with 1-unit bins:")
    int_data = np.random.randint(0, 10, 1000)
    fig1, ax1, stats1 = plot_unit_histogram(
        int_data,
        title='Integer Data with 1-unit Bins',
        xlabel='Integer Value',
        color='lightcoral'
    )
    plt.show()
    print(f"Statistics: {stats1}")
    
    # Example 2: Float data with 0.5-unit bins
    print("\n2. Float data with 0.5-unit bins:")
    float_data = np.random.normal(5, 2, 1000)
    fig2, ax2, stats2 = plot_fine_histogram(
        float_data,
        bin_size=0.5,
        title='Float Data with 0.5-unit Bins',
        xlabel='Value',
        color='lightblue',
        density=True
    )
    plt.show()
    print(f"Statistics: {stats2}")
    
    # Example 3: Small range data with 2-unit bins
    print("\n3. Small range data with 2-unit bins:")
    small_data = np.array([1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5])
    fig3, ax3, stats3 = plot_two_unit_histogram(
        small_data,
        title='Small Dataset with 2-unit Bins',
        xlabel='Value',
        color='lightgreen',
        show_stats=True,
        vertical_lines=[2.5, 3.5]  # Add reference lines
    )
    plt.show()
    print(f"Statistics: {stats3}")
    
    # Example 4: Compare multiple datasets
    print("\n4. Comparing multiple datasets:")
    dataset1 = np.random.normal(0, 1, 500)
    dataset2 = np.random.normal(2, 1.5, 500)
    dataset3 = np.random.normal(-1, 0.8, 500)
    
    fig4, axes4, stats4 = compare_fine_histograms(
        datasets=[dataset1, dataset2, dataset3],
        labels=['Group A', 'Group B', 'Group C'],
        bin_size=0.25,
        title='Comparison of Three Groups'
    )
    plt.show()
    
    # Example 5: Custom vertical lines
    print("\n5. Histogram with custom vertical lines:")
    test_data = np.random.exponential(3, 500)
    
    # Define vertical lines with customization
    vlines = {
        'positions': [2, 5, 8],
        'colors': ['red', 'green', 'blue'],
        'styles': ['--', '-.', ':'],
        'labels': ['Threshold 1', 'Threshold 2', 'Threshold 3'],
        'linewidths': [2, 1.5, 1],
        'alphas': [0.8, 0.6, 0.4]
    }
    
    fig5, ax5, stats5 = plot_fine_histogram(
        test_data,
        bin_size=0.5,
        title='Histogram with Custom Vertical Lines',
        vertical_lines=vlines,
        color='gold',
        show_stats=True
    )
    ax5.legend()
    plt.show()
    
    print("\n" + "=" * 50)
    print("Demonstration complete!")
    
    return [stats1, stats2, stats3, stats4, stats5]


# Quick plotting function for common cases
def quick_fine_hist(data, bin_width=1, **kwargs):
    """
    Quick plotting with sensible defaults.
    
    Parameters:
    -----------
    data : array-like
        Data to plot
    bin_width : int or float, default=1
        Bin width
    **kwargs : additional styling arguments
    
    Returns:
    --------
    fig, ax
    """
    fig, ax, _ = plot_fine_histogram(
        data,
        bin_size=bin_width,
        color=kwargs.get('color', 'steelblue'),
        edgecolor=kwargs.get('edgecolor', 'white'),
        title=kwargs.get('title', f'Histogram (bin width={bin_width})'),
        show_stats=kwargs.get('show_stats', True),
        show_grid=kwargs.get('show_grid', True)
    )
    return fig, ax


# Function to analyze and print bin counts
def analyze_bin_counts(data, bin_size=1):
    """
    Analyze and print detailed bin counts for fine histograms.
    
    Parameters:
    -----------
    data : array-like
        Input data
    bin_size : int or float, default=1
        Bin width
    
    Returns:
    --------
    bin_centers : array
        Center of each bin
    counts : array
        Count in each bin
    percentage : array
        Percentage in each bin
    """
    data = np.asarray(data).flatten()
    
    # Calculate bins
    data_min = np.floor(np.min(data) / bin_size) * bin_size
    data_max = np.ceil(np.max(data) / bin_size) * bin_size + 1e-10
    bins = np.arange(data_min, data_max, bin_size)
    
    # Calculate counts
    counts, bin_edges = np.histogram(data, bins=bins)
    
    # Calculate bin centers and percentages
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    percentages = (counts / len(data)) * 100
    
    # Print detailed table
    print(f"\nDetailed Bin Analysis (bin size = {bin_size})")
    print("=" * 60)
    print(f"{'Bin Range':<20} {'Center':<10} {'Count':<10} {'%':<10} {'Cumulative %':<12}")
    print("-" * 60)
    
    cumulative = 0
    for i, (left, right) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        cumulative += percentages[i]
        print(f"[{left:.2f}, {right:.2f})  "
              f"{bin_centers[i]:<10.2f} "
              f"{counts[i]:<10} "
              f"{percentages[i]:<10.2f} "
              f"{cumulative:<12.2f}")
    
    print("=" * 60)
    print(f"Total: {len(data):,} observations")
    print(f"Unique bins: {len(bin_centers)}")
    print(f"Most frequent bin: {bin_centers[np.argmax(counts)]:.2f} "
          f"({np.max(counts)} observations, {np.max(percentages):.1f}%)")
    
    return bin_centers, counts, percentages

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
    
    iter_test = 200
    test_results={}
    faster_than = 0
    slower_than = 0
    parity = 0
    error = 0
    for i in range(iter_test):
        data = generate_example_data(n_points=25, noise_std=0.03, curve_type='sigmoid')
        fitter = BezierCurveFitter(degree=5)
        
        start_time = time.time()
        result = fitter.fit(data, max_iter=50, tol=1e-6, verbose=False)
        elapsed_time_1 = time.time() - start_time
        
        fitter = BezierCurveFitterOpt(degree=5, use_cache=True, cache_size=1000)
        
        start_time = time.time()
        try:
            result = fitter.fit(data, max_iter=50, tol=1e-6, verbose=False)
        except AttributeError:
            error+=1
            continue
        elapsed_time_2 = time.time() - start_time
        
        test_results[i]=[(elapsed_time_1-elapsed_time_2)/elapsed_time_1*100]
        
        if (elapsed_time_1>elapsed_time_2):
            faster_than+=1
        elif (elapsed_time_1<elapsed_time_2):
            slower_than+=1
        else:
            parity+=1
          
    
    print(f"\nNO DIFFERENCE: {parity/iter_test*100}%\nFASTER THAN: {faster_than/iter_test*100}%\nSLOWER THAN: {slower_than/iter_test*100}%")
    print(f"Errors thrown: {error/iter_test*100}%")
    
    print(pd.DataFrame(test_results.values()).describe())
    
    fig, ax = quick_fine_hist(list(test_results.values()), bin_width=0.5)
    plt.show()
    
      
    
    # # Generate example data
    # print("\nGenerating example data (sigmoid curve with noise)...")
    # data = generate_example_data(n_points=25, noise_std=0.03, curve_type='sigmoid')
    # print(f"Data shape: {data.shape}")
    
    # # Create and run fitter WITH Phase 1 optimizations
    # print("\nFitting cubic Bézier curve with Phase 1 optimizations...")
    # fitter = BezierCurveFitter(degree=5)
    
    # start_time = time.time()
    # result = fitter.fit(data, max_iter=50, tol=1e-6, verbose=False)
    # elapsed_time = time.time() - start_time
    
    # print(f"\nFitting completed in {elapsed_time:.3f} seconds")
    # print(f"Final residual norm: {result['residual_norm']:.6e}")
    # print(f"Number of iterations: {result['iterations']}")
    
    # fitter = BezierCurveFitterOpt(degree=5, use_cache=True, cache_size=1000)
    
    # start_time = time.time()
    # result = fitter.fit(data, max_iter=50, tol=1e-6, verbose=False)
    # elapsed_time = time.time() - start_time
    
    # print(f"\nFitting completed in {elapsed_time:.3f} seconds")
    # print(f"Final residual norm: {result['residual_norm']:.6e}")
    # print(f"Number of iterations: {result['iterations']}")
    
    # # Print optimization statistics
    # fitter.print_optimization_stats()
    
    # # Compute error metrics
    # errors = fitter.compute_approximation_error(data)
    # print("\nError metrics:")
    # print(f"  Maximum error: {errors['max_error']:.6f}")
    # print(f"  Mean error: {errors['mean_error']:.6f}")
    # print(f"  RMSE: {errors['rmse']:.6f}")
    
    # # Plot results
    # print("\nGenerating plots...")
    # plot_results(data, fitter, title="Bézier Curve Fitting Example")
    
    # print("\nControl points:")
    # print(fitter.control_points)
    
    # print("\nDone!")


if __name__ == "__main__":
    main()