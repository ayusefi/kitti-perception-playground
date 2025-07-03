#!/usr/bin/env python3
"""
KITTI Lidar-to-Camera Projection Script

This script projects 3D LiDAR points onto 2D camera images using KITTI calibration data.
It demonstrates sensor fusion between LiDAR and camera data for autonomous vehicle perception.

Dependencies:
- numpy
- opencv-python
- matplotlib (optional, for colormap)

Usage:
    python project_lidar_to_camera.py
"""

import numpy as np
import cv2
import os
import glob
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class KittiProjector:
    """
    KITTI LiDAR-to-Camera projection class
    
    Handles loading calibration data, synchronizing timestamps, and projecting
    3D LiDAR points onto 2D camera images.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the projector with KITTI data path
        
        Args:
            data_path: Path to KITTI dataset (e.g., "2011_09_26_drive_0001_sync")
        """
        self.data_path = data_path
        self.calib_data = {}
        self.timestamps_cam = []
        self.timestamps_lidar = []
        
        # Load calibration and timestamp data
        self._load_calibration()
        self._load_timestamps()
    
    def _load_calibration(self) -> None:
        """Load calibration matrices from KITTI calibration files"""
        calib_file = os.path.join(self.data_path, "calib_cam_to_cam.txt")
        
        if not os.path.exists(calib_file):
            raise FileNotFoundError(f"Calibration file not found: {calib_file}")
        
        with open(calib_file, 'r') as f:
            lines = f.readlines()
        
        # Parse calibration data
        for line in lines:
            if line.startswith('P_rect_02:'):  # Camera 2 projection matrix
                self.calib_data['P2'] = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
            elif line.startswith('R_rect_02:'):  # Camera 2 rectification matrix
                self.calib_data['R2'] = np.array([float(x) for x in line.split()[1:]]).reshape(3, 3)
        
        # Load Velodyne to camera calibration
        calib_velo_file = os.path.join(self.data_path, "calib_velo_to_cam.txt")
        if os.path.exists(calib_velo_file):
            with open(calib_velo_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                if line.startswith('R:'):
                    self.calib_data['R_velo_to_cam'] = np.array([float(x) for x in line.split()[1:]]).reshape(3, 3)
                elif line.startswith('T:'):
                    self.calib_data['T_velo_to_cam'] = np.array([float(x) for x in line.split()[1:]]).reshape(3, 1)
        
        print("Calibration data loaded successfully")
    
    def _load_timestamps(self) -> None:
        """Load timestamp data for camera and LiDAR"""
        # Load camera timestamps
        cam_timestamp_file = os.path.join(self.data_path, "image_02/timestamps.txt")
        if os.path.exists(cam_timestamp_file):
            with open(cam_timestamp_file, 'r') as f:
                self.timestamps_cam = [line.strip() for line in f.readlines()]
        
        # Load LiDAR timestamps
        lidar_timestamp_file = os.path.join(self.data_path, "velodyne_points/timestamps.txt")
        if os.path.exists(lidar_timestamp_file):
            with open(lidar_timestamp_file, 'r') as f:
                self.timestamps_lidar = [line.strip() for line in f.readlines()]
        
        print(f"Loaded {len(self.timestamps_cam)} camera timestamps and {len(self.timestamps_lidar)} LiDAR timestamps")
    
    def load_lidar_points(self, frame_idx: int) -> np.ndarray:
        """
        Load LiDAR point cloud for a specific frame
        
        Args:
            frame_idx: Frame index
            
        Returns:
            numpy array of shape (N, 4) with [x, y, z, intensity]
        """
        lidar_files = sorted(glob.glob(os.path.join(self.data_path, "velodyne_points/data/*.bin")))
        
        if frame_idx >= len(lidar_files):
            raise IndexError(f"Frame index {frame_idx} out of range (max: {len(lidar_files)-1})")
        
        # Load binary point cloud data
        points = np.fromfile(lidar_files[frame_idx], dtype=np.float32).reshape(-1, 4)
        return points
    
    def load_camera_image(self, frame_idx: int) -> np.ndarray:
        """
        Load camera image for a specific frame
        
        Args:
            frame_idx: Frame index
            
        Returns:
            numpy array representing the image
        """
        image_files = sorted(glob.glob(os.path.join(self.data_path, "image_02/data/*.png")))
        
        if frame_idx >= len(image_files):
            raise IndexError(f"Frame index {frame_idx} out of range (max: {len(image_files)-1})")
        
        image = cv2.imread(image_files[frame_idx])
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def project_lidar_to_camera(self, lidar_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 3D LiDAR points to 2D camera coordinates
        
        Args:
            lidar_points: LiDAR points array of shape (N, 4) [x, y, z, intensity]
            
        Returns:
            Tuple of (projected_points, depths) where:
            - projected_points: (N, 2) array of [u, v] pixel coordinates
            - depths: (N,) array of depth values
        """
        # Extract XYZ coordinates (ignore intensity)
        points_3d = lidar_points[:, :3]
        
        # Convert to homogeneous coordinates
        points_3d_homo = np.column_stack([points_3d, np.ones(points_3d.shape[0])])
        
        # Transform from Velodyne to camera coordinate system
        if 'R_velo_to_cam' in self.calib_data and 'T_velo_to_cam' in self.calib_data:
            # Create transformation matrix
            T_velo_to_cam = np.eye(4)
            T_velo_to_cam[:3, :3] = self.calib_data['R_velo_to_cam']
            T_velo_to_cam[:3, 3:4] = self.calib_data['T_velo_to_cam']
            
            # Transform points
            points_cam = (T_velo_to_cam @ points_3d_homo.T).T
        else:
            # If no Velodyne calibration, assume points are already in camera frame
            points_cam = points_3d_homo
        
        # Apply rectification if available
        if 'R2' in self.calib_data:
            R_rect = np.eye(4)
            R_rect[:3, :3] = self.calib_data['R2']
            points_cam = (R_rect @ points_cam.T).T
        
        # Project to image plane using camera projection matrix
        if 'P2' in self.calib_data:
            # Project 3D points to 2D
            points_2d_homo = (self.calib_data['P2'] @ points_cam[:, :4].T).T
            
            # Convert from homogeneous to Cartesian coordinates
            points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
            depths = points_cam[:, 2]  # Z coordinate is depth
            
            return points_2d, depths
        else:
            raise ValueError("Camera projection matrix P2 not found in calibration data")
    
    def filter_points_in_image(self, points_2d: np.ndarray, depths: np.ndarray, 
                             image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter points that are within the image bounds and have positive depth
        
        Args:
            points_2d: 2D projected points
            depths: Corresponding depth values
            image_shape: (height, width) of the image
            
        Returns:
            Tuple of filtered (points_2d, depths)
        """
        height, width = image_shape[:2]
        
        # Create mask for valid points
        valid_mask = (
            (points_2d[:, 0] >= 0) & (points_2d[:, 0] < width) &
            (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height) &
            (depths > 0)  # Only points in front of camera
        )
        
        return points_2d[valid_mask], depths[valid_mask]
    
    def visualize_projection(self, image: np.ndarray, points_2d: np.ndarray, 
                           depths: np.ndarray, max_depth: float = 50.0) -> np.ndarray:
        """
        Visualize LiDAR points projected onto camera image
        
        Args:
            image: Camera image
            points_2d: 2D projected points
            depths: Corresponding depth values
            max_depth: Maximum depth for color mapping
            
        Returns:
            Image with LiDAR points overlaid
        """
        # Create a copy of the image
        result_image = image.copy()
        
        # Normalize depths for color mapping
        normalized_depths = np.clip(depths / max_depth, 0, 1)
        
        # Create colormap (closer points = red, farther points = blue)
        colors = plt.cm.jet(1 - normalized_depths)  # Invert so red = close, blue = far
        
        # Draw points on image
        for i, (point, color) in enumerate(zip(points_2d, colors)):
            x, y = int(point[0]), int(point[1])
            
            # Convert matplotlib color to OpenCV BGR
            bgr_color = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
            
            # Draw point
            cv2.circle(result_image, (x, y), 2, bgr_color, -1)
        
        return result_image

def main():
    """Main function to demonstrate LiDAR-to-Camera projection"""
    
    # Configuration
    data_path = "data/2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync"  # Adjust this path
    frame_idx = 10  # Frame to process
    output_dir = "output"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize projector
        print("Initializing KITTI projector...")
        projector = KittiProjector(data_path)
        
        # Load data
        print(f"Loading frame {frame_idx}...")
        lidar_points = projector.load_lidar_points(frame_idx)
        camera_image = projector.load_camera_image(frame_idx)
        
        print(f"Loaded {len(lidar_points)} LiDAR points and camera image of shape {camera_image.shape}")
        
        # Project LiDAR points to camera
        print("Projecting LiDAR points to camera...")
        points_2d, depths = projector.project_lidar_to_camera(lidar_points)
        
        # Filter points within image bounds
        points_2d_filtered, depths_filtered = projector.filter_points_in_image(
            points_2d, depths, camera_image.shape
        )
        
        print(f"Filtered to {len(points_2d_filtered)} valid points within image bounds")
        
        # Create visualization
        print("Creating visualization...")
        result_image = projector.visualize_projection(
            camera_image, points_2d_filtered, depths_filtered
        )
        
        # Save results
        output_path = os.path.join(output_dir, f"lidar_projection_frame_{frame_idx:06d}.png")
        
        # Convert RGB to BGR for OpenCV saving
        result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_bgr)
        
        print(f"Saved projection result to: {output_path}")
        
        # Display result (optional)
        cv2.imshow("LiDAR-to-Camera Projection", result_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Print statistics
        print("\nProjection Statistics:")
        print(f"  Total LiDAR points: {len(lidar_points)}")
        print(f"  Points in image: {len(points_2d_filtered)}")
        print(f"  Depth range: {depths_filtered.min():.2f} - {depths_filtered.max():.2f} meters")
        print(f"  Average depth: {depths_filtered.mean():.2f} meters")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the KITTI dataset is properly downloaded and the path is correct.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()