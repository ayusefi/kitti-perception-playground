#!/usr/bin/env python3
"""
Kalman Filter Implementation for 3D Object Tracking

This module implements a generic Kalman Filter class suitable for tracking
3D objects with constant velocity model. The state vector includes position
and velocity: [x, y, z, vx, vy, vz]

Author: Abdullah Yusefi
Date: July 2025
"""

import numpy as np
from typing import Optional, Tuple


class KalmanFilter:
    """
    Generic Kalman Filter implementation for 3D object tracking
    
    State vector: [x, y, z, vx, vy, vz] (position and velocity)
    Measurement vector: [x, y, z] (position only)
    
    The filter assumes a constant velocity model with Gaussian noise.
    """
    
    def __init__(self, dt: float = 0.1, process_noise: float = 1.0, measurement_noise: float = 1.0):
        """
        Initialize the Kalman Filter
        
        Args:
            dt: Time step between predictions (seconds)
            process_noise: Process noise variance (affects prediction uncertainty)
            measurement_noise: Measurement noise variance (affects measurement trust)
        """
        self.dt = dt
        self.state_dim = 6  # [x, y, z, vx, vy, vz]
        self.measurement_dim = 3  # [x, y, z]
        
        # State vector [x, y, z, vx, vy, vz]
        self.x = np.zeros((self.state_dim, 1))
        
        # State covariance matrix
        self.P = np.eye(self.state_dim) * 1000  # High initial uncertainty
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix (we only measure position)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Process noise covariance matrix
        self.Q = self._build_process_noise_matrix(process_noise)
        
        # Measurement noise covariance matrix
        self.R = np.eye(self.measurement_dim) * measurement_noise
        
        # Identity matrix for calculations
        self.I = np.eye(self.state_dim)
        
        # Track filter statistics
        self.prediction_count = 0
        self.update_count = 0
        
    def _build_process_noise_matrix(self, noise_variance: float) -> np.ndarray:
        """
        Build the process noise covariance matrix Q
        
        Uses a continuous white noise acceleration model
        
        Args:
            noise_variance: Base noise variance
            
        Returns:
            Process noise covariance matrix
        """
        dt = self.dt
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        
        # Noise affects both position and velocity
        q_pos = noise_variance * dt4 / 4
        q_vel = noise_variance * dt2
        q_pos_vel = noise_variance * dt3 / 2
        
        Q = np.array([
            [q_pos, 0, 0, q_pos_vel, 0, 0],
            [0, q_pos, 0, 0, q_pos_vel, 0],
            [0, 0, q_pos, 0, 0, q_pos_vel],
            [q_pos_vel, 0, 0, q_vel, 0, 0],
            [0, q_pos_vel, 0, 0, q_vel, 0],
            [0, 0, q_pos_vel, 0, 0, q_vel]
        ])
        
        return Q
    
    def initialize(self, initial_position: np.ndarray, initial_velocity: Optional[np.ndarray] = None) -> None:
        """
        Initialize the filter with an initial state
        
        Args:
            initial_position: Initial position [x, y, z]
            initial_velocity: Initial velocity [vx, vy, vz] (defaults to zero)
        """
        if initial_velocity is None:
            initial_velocity = np.zeros(3)
            
        self.x = np.array([
            [initial_position[0]],
            [initial_position[1]],
            [initial_position[2]],
            [initial_velocity[0]],
            [initial_velocity[1]],
            [initial_velocity[2]]
        ])
        
        # Reset covariance matrix
        self.P = np.eye(self.state_dim) * 1000
        self.P[3:6, 3:6] *= 100  # Lower uncertainty for velocity
        
        # Reset counters
        self.prediction_count = 0
        self.update_count = 0
        
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the next state using the motion model
        
        Returns:
            Tuple of (predicted_state, predicted_covariance)
        """
        # Predict state: x = F * x
        self.x = self.F @ self.x
        
        # Predict covariance: P = F * P * F' + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        self.prediction_count += 1
        
        return self.x.copy(), self.P.copy()
    
    def update(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update the filter with a new measurement
        
        Args:
            measurement: Measurement vector [x, y, z]
            
        Returns:
            Tuple of (updated_state, updated_covariance)
        """
        # Ensure measurement is column vector
        z = measurement.reshape(-1, 1)
        
        # Innovation (residual): y = z - H * x
        y = z - self.H @ self.x
        
        # Innovation covariance: S = H * P * H' + R
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain: K = P * H' * S^(-1)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state: x = x + K * y
        self.x = self.x + K @ y
        
        # Update covariance: P = (I - K * H) * P
        self.P = (self.I - K @ self.H) @ self.P
        
        self.update_count += 1
        
        return self.x.copy(), self.P.copy()
    
    def get_position(self) -> np.ndarray:
        """
        Get current position estimate
        
        Returns:
            Position vector [x, y, z]
        """
        return self.x[:3].flatten()
    
    def get_velocity(self) -> np.ndarray:
        """
        Get current velocity estimate
        
        Returns:
            Velocity vector [vx, vy, vz]
        """
        return self.x[3:6].flatten()
    
    def get_position_uncertainty(self) -> np.ndarray:
        """
        Get position uncertainty (standard deviation)
        
        Returns:
            Position uncertainty [σx, σy, σz]
        """
        return np.sqrt(np.diag(self.P[:3, :3]))
    
    def get_velocity_uncertainty(self) -> np.ndarray:
        """
        Get velocity uncertainty (standard deviation)
        
        Returns:
            Velocity uncertainty [σvx, σvy, σvz]
        """
        return np.sqrt(np.diag(self.P[3:6, 3:6]))
    
    def get_innovation_distance(self, measurement: np.ndarray) -> float:
        """
        Calculate Mahalanobis distance for data association
        
        Args:
            measurement: Measurement vector [x, y, z]
            
        Returns:
            Mahalanobis distance
        """
        z = measurement.reshape(-1, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        
        # Mahalanobis distance
        distance = np.sqrt(y.T @ np.linalg.inv(S) @ y)
        return float(distance)
    
    def get_euclidean_distance(self, measurement: np.ndarray) -> float:
        """
        Calculate Euclidean distance to measurement
        
        Args:
            measurement: Measurement vector [x, y, z]
            
        Returns:
            Euclidean distance
        """
        current_pos = self.get_position()
        return np.linalg.norm(measurement - current_pos)
    
    def get_state_dict(self) -> dict:
        """
        Get current state as a dictionary for easy access
        
        Returns:
            Dictionary containing state information
        """
        return {
            'position': self.get_position(),
            'velocity': self.get_velocity(),
            'position_uncertainty': self.get_position_uncertainty(),
            'velocity_uncertainty': self.get_velocity_uncertainty(),
            'prediction_count': self.prediction_count,
            'update_count': self.update_count
        }
    
    def __str__(self) -> str:
        """String representation of the filter state"""
        pos = self.get_position()
        vel = self.get_velocity()
        return f"KF: pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], vel=[{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}]"


def test_kalman_filter():
    """
    Test the Kalman Filter with a simple simulation
    """
    print("Testing Kalman Filter...")
    
    # Initialize filter
    kf = KalmanFilter(dt=0.1, process_noise=0.1, measurement_noise=0.5)
    
    # Initial state: object at origin moving in +x direction
    initial_pos = np.array([0.0, 0.0, 0.0])
    initial_vel = np.array([1.0, 0.0, 0.0])
    kf.initialize(initial_pos, initial_vel)
    
    print(f"Initial state: {kf}")
    
    # Simulate 10 time steps
    true_positions = []
    estimated_positions = []
    
    for i in range(10):
        # Predict
        kf.predict()
        
        # Generate noisy measurement
        true_pos = np.array([i * 0.1, 0.0, 0.0])  # True position
        noise = np.random.normal(0, 0.1, 3)  # Measurement noise
        measurement = true_pos + noise
        
        # Update
        kf.update(measurement)
        
        # Store results
        true_positions.append(true_pos)
        estimated_positions.append(kf.get_position())
        
        print(f"Step {i+1}: True pos={true_pos}, Estimated pos={kf.get_position()}")
    
    # Calculate RMS error
    errors = [np.linalg.norm(true - est) for true, est in zip(true_positions, estimated_positions)]
    rms_error = np.sqrt(np.mean(np.array(errors)**2))
    print(f"RMS error: {rms_error:.4f}")
    
    print("Test completed successfully!")


if __name__ == "__main__":
    test_kalman_filter()