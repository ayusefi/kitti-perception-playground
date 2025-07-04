#!/usr/bin/env python3
"""
KITTI Ground Plane Segmentation Script using RANSAC

This script implements ground plane removal from 3D LiDAR point clouds using the RANSAC
(Random Sample Consensus) algorithm. This is a fundamental preprocessing step in
autonomous vehicle perception to isolate objects of interest from the ground.

Dependencies:
- numpy
- open3d
- matplotlib

Usage:
    python segment_ground.py
"""

import numpy as np
import open3d as o3d
import os
import glob
from typing import Tuple, Optional
import matplotlib.pyplot as plt

class GroundSegmenter:
    """
    Ground plane segmentation using RANSAC algorithm
    
    This class provides functionality to:
    1. Load 3D point clouds from KITTI dataset
    2. Apply RANSAC algorithm for plane detection
    3. Segment ground points from object points
    4. Visualize and save results
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the ground segmenter
        
        Args:
            data_path: Path to KITTI dataset directory
        """
        self.data_path = data_path
        self.ransac_params = {
            'distance_threshold': 0.2,    # Points within 20cm of plane are considered inliers
            'ransac_n': 3,                # Minimum points needed to define a plane
            'num_iterations': 1000        # Maximum RANSAC iterations
        }
    
    def load_point_cloud(self, frame_idx: int) -> o3d.geometry.PointCloud:
        """
        Load a point cloud from KITTI dataset
        
        Args:
            frame_idx: Index of the frame to load
            
        Returns:
            Open3D PointCloud object
        """
        # Find all LiDAR files
        lidar_files = sorted(glob.glob(os.path.join(self.data_path, "velodyne_points/data/*.bin")))
        
        if frame_idx >= len(lidar_files):
            raise IndexError(f"Frame index {frame_idx} out of range (max: {len(lidar_files)-1})")
        
        # Load binary point cloud data
        points = np.fromfile(lidar_files[frame_idx], dtype=np.float32).reshape(-1, 4)
        
        # Extract XYZ coordinates (ignore intensity for ground segmentation)
        xyz_points = points[:, :3]
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_points)
        
        return pcd
    
    def segment_ground_plane(self, pcd: o3d.geometry.PointCloud) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment ground plane using RANSAC algorithm
        
        Args:
            pcd: Input point cloud
            
        Returns:
            Tuple of (plane_model, ground_indices, object_indices)
            - plane_model: [a, b, c, d] coefficients of plane equation ax + by + cz + d = 0
            - ground_indices: Indices of points belonging to ground
            - object_indices: Indices of points belonging to objects
        """
        print("Applying RANSAC for ground plane detection...")
        
        # Apply RANSAC algorithm using Open3D
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=self.ransac_params['distance_threshold'],
            ransac_n=self.ransac_params['ransac_n'],
            num_iterations=self.ransac_params['num_iterations']
        )
        
        # Convert to numpy arrays
        ground_indices = np.array(inliers)
        all_indices = np.arange(len(pcd.points))
        object_indices = np.setdiff1d(all_indices, ground_indices)
        
        # Print plane equation
        a, b, c, d = plane_model
        print(f"Ground plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
        print(f"Ground points: {len(ground_indices)}")
        print(f"Object points: {len(object_indices)}")
        print(f"Ground ratio: {len(ground_indices)/len(pcd.points)*100:.1f}%")
        
        return plane_model, ground_indices, object_indices
    
    def create_segmented_visualization(self, pcd: o3d.geometry.PointCloud, 
                                     ground_indices: np.ndarray, 
                                     object_indices: np.ndarray) -> o3d.geometry.PointCloud:
        """
        Create a colored visualization of ground vs object segmentation
        
        Args:
            pcd: Original point cloud
            ground_indices: Indices of ground points
            object_indices: Indices of object points
            
        Returns:
            Colored point cloud for visualization
        """
        # Create a copy of the point cloud for visualization
        vis_pcd = o3d.geometry.PointCloud()
        vis_pcd.points = pcd.points
        
        # Initialize colors array
        colors = np.zeros((len(pcd.points), 3))
        
        # Color ground points gray [0.5, 0.5, 0.5]
        colors[ground_indices] = [0.5, 0.5, 0.5]
        
        # Color object points red [1.0, 0.0, 0.0]
        colors[object_indices] = [1.0, 0.0, 0.0]
        
        # Apply colors to point cloud
        vis_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return vis_pcd
    
    def analyze_ground_plane(self, plane_model: np.ndarray, pcd: o3d.geometry.PointCloud, 
                           ground_indices: np.ndarray) -> dict:
        """
        Analyze the detected ground plane properties
        
        Args:
            plane_model: Plane equation coefficients [a, b, c, d]
            pcd: Original point cloud
            ground_indices: Indices of ground points
            
        Returns:
            Dictionary containing analysis results
        """
        a, b, c, d = plane_model
        
        # Calculate plane normal vector
        normal = np.array([a, b, c])
        normal_normalized = normal / np.linalg.norm(normal)
        
        # Calculate angle with horizontal plane (XY plane normal is [0, 0, 1])
        horizontal_normal = np.array([0, 0, 1])
        angle_rad = np.arccos(np.abs(np.dot(normal_normalized, horizontal_normal)))
        angle_deg = np.degrees(angle_rad)
        
        # Calculate ground points statistics
        ground_points = np.asarray(pcd.points)[ground_indices]
        
        analysis = {
            'plane_equation': plane_model,
            'normal_vector': normal_normalized,
            'angle_with_horizontal': angle_deg,
            'ground_height_mean': np.mean(ground_points[:, 2]),
            'ground_height_std': np.std(ground_points[:, 2]),
            'ground_points_count': len(ground_indices),
            'total_points_count': len(pcd.points),
            'ground_ratio': len(ground_indices) / len(pcd.points)
        }
        
        return analysis
    
    def save_visualization(self, vis_pcd: o3d.geometry.PointCloud, 
                          output_path: str, 
                          view_angle: str = "side") -> None:
        """
        Save visualization of segmented point cloud
        
        Args:
            vis_pcd: Colored point cloud
            output_path: Path to save the image
            view_angle: Camera view angle ("side", "top", "front")
        """
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)  # Don't show window
        vis.add_geometry(vis_pcd)
        
        # Set view angle
        ctr = vis.get_view_control()
        
        if view_angle == "side":
            # Side view - good for seeing ground plane
            ctr.set_lookat([0, 0, 0])
            ctr.set_up([0, 0, 1])
            ctr.set_front([1, 0, 0])
        elif view_angle == "top":
            # Top view - bird's eye view
            ctr.set_lookat([0, 0, 0])
            ctr.set_up([0, 1, 0])
            ctr.set_front([0, 0, -1])
        elif view_angle == "front":
            # Front view
            ctr.set_lookat([0, 0, 0])
            ctr.set_up([0, 0, 1])
            ctr.set_front([0, 1, 0])
        
        # Set camera distance
        ctr.set_zoom(0.3)
        
        # Render and save
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(output_path)
        vis.destroy_window()
        
        print(f"Saved visualization to: {output_path}")
    
    def visualize_interactive(self, vis_pcd: o3d.geometry.PointCloud) -> None:
        """
        Show interactive visualization of segmented point cloud
        
        Args:
            vis_pcd: Colored point cloud
        """
        print("\nOpening interactive visualization...")
        print("Controls:")
        print("- Mouse: Rotate view")
        print("- Scroll: Zoom in/out")
        print("- Shift+Mouse: Pan")
        print("- Press 'q' to quit")
        
        # Create custom visualization with better lighting
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Ground Plane Segmentation", width=1200, height=800)
        vis.add_geometry(vis_pcd)
        
        # Improve rendering options
        opt = vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
        opt.point_size = 2.0
        opt.light_on = True
        
        # Set good initial view
        ctr = vis.get_view_control()
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
        ctr.set_front([1, 0, 0])
        ctr.set_zoom(0.3)
        
        # Run visualization
        vis.run()
        vis.destroy_window()

def print_ransac_explanation():
    """Print educational explanation of RANSAC algorithm"""
    print("\n" + "="*60)
    print("RANSAC (Random Sample Consensus) Algorithm Explanation")
    print("="*60)
    print("""
RANSAC is a robust algorithm for fitting models to data with outliers.
For ground plane detection, it works as follows:

1. HYPOTHESIS GENERATION:
   - Randomly select 3 points from the point cloud
   - Fit a plane through these 3 points
   - Calculate plane equation: ax + by + cz + d = 0

2. CONSENSUS EVALUATION:
   - For each remaining point, calculate distance to the plane
   - If distance < threshold, count as 'inlier' (supports the hypothesis)
   - If distance >= threshold, count as 'outlier'

3. MODEL SELECTION:
   - Repeat steps 1-2 for many iterations (e.g., 1000 times)
   - Keep track of the plane with the most inliers
   - The plane with maximum inliers is the final ground plane

4. ADVANTAGES:
   - Robust to outliers (cars, trees, buildings)
   - Works well when ground is the dominant plane
   - Computationally efficient

5. PARAMETERS:
   - distance_threshold: How close a point must be to be considered ground
   - num_iterations: How many random samples to try
   - ransac_n: Minimum points needed (3 for a plane)
""")
    print("="*60)

def main():
    """Main function to demonstrate ground plane segmentation"""
    
    # Configuration
    data_path = "data/2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync"  # Adjust this path
    frame_idx = 10  # Frame to process
    output_dir = "output"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Print educational content
    print_ransac_explanation()
    
    try:
        # Initialize ground segmenter
        print("\nInitializing Ground Segmenter...")
        segmenter = GroundSegmenter(data_path)
        
        # Load point cloud
        print(f"\nLoading point cloud for frame {frame_idx}...")
        pcd = segmenter.load_point_cloud(frame_idx)
        print(f"Loaded point cloud with {len(pcd.points)} points")
        
        # Segment ground plane
        print(f"\nApplying RANSAC with parameters:")
        print(f"  Distance threshold: {segmenter.ransac_params['distance_threshold']} meters")
        print(f"  RANSAC iterations: {segmenter.ransac_params['num_iterations']}")
        print(f"  Min points for plane: {segmenter.ransac_params['ransac_n']}")
        
        plane_model, ground_indices, object_indices = segmenter.segment_ground_plane(pcd)
        
        # Create visualization
        print("\nCreating colored visualization...")
        vis_pcd = segmenter.create_segmented_visualization(pcd, ground_indices, object_indices)
        
        # Analyze results
        print("\nAnalyzing ground plane properties...")
        analysis = segmenter.analyze_ground_plane(plane_model, pcd, ground_indices)
        
        # Print analysis results
        print("\nGround Plane Analysis:")
        print(f"  Plane equation: {analysis['plane_equation'][0]:.4f}x + {analysis['plane_equation'][1]:.4f}y + {analysis['plane_equation'][2]:.4f}z + {analysis['plane_equation'][3]:.4f} = 0")
        print(f"  Normal vector: [{analysis['normal_vector'][0]:.4f}, {analysis['normal_vector'][1]:.4f}, {analysis['normal_vector'][2]:.4f}]")
        print(f"  Angle with horizontal: {analysis['angle_with_horizontal']:.2f}°")
        print(f"  Ground height (mean): {analysis['ground_height_mean']:.2f} ± {analysis['ground_height_std']:.2f} meters")
        print(f"  Ground points: {analysis['ground_points_count']:,} ({analysis['ground_ratio']*100:.1f}%)")
        print(f"  Object points: {analysis['total_points_count'] - analysis['ground_points_count']:,}")
        
        # Save visualizations from different angles
        print("\nSaving visualizations...")
        
        # Side view
        side_view_path = os.path.join(output_dir, f"ground_segmentation_side_frame_{frame_idx:06d}.png")
        segmenter.save_visualization(vis_pcd, side_view_path, "side")
        
        # Top view
        top_view_path = os.path.join(output_dir, f"ground_segmentation_top_frame_{frame_idx:06d}.png")
        segmenter.save_visualization(vis_pcd, top_view_path, "top")
        
        # Front view
        front_view_path = os.path.join(output_dir, f"ground_segmentation_front_frame_{frame_idx:06d}.png")
        segmenter.save_visualization(vis_pcd, front_view_path, "front")
        
        # Create summary plot
        print("\nCreating analysis summary plot...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Ground Plane Segmentation Analysis', fontsize=16)
        
        # Histogram of ground point heights
        ground_points = np.asarray(pcd.points)[ground_indices]
        axes[0, 0].hist(ground_points[:, 2], bins=50, alpha=0.7, color='gray')
        axes[0, 0].set_title('Ground Point Height Distribution')
        axes[0, 0].set_xlabel('Height (meters)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Pie chart of ground vs objects
        sizes = [len(ground_indices), len(object_indices)]
        labels = ['Ground', 'Objects']
        colors = ['lightgray', 'red']
        axes[0, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        axes[0, 1].set_title('Point Classification')
        
        # Scatter plot of ground plane fit
        sample_indices = np.random.choice(len(ground_indices), min(1000, len(ground_indices)), replace=False)
        sample_ground = ground_points[sample_indices]
        axes[1, 0].scatter(sample_ground[:, 0], sample_ground[:, 1], 
                          c=sample_ground[:, 2], cmap='viridis', alpha=0.6, s=1)
        axes[1, 0].set_title('Ground Points (Top View)')
        axes[1, 0].set_xlabel('X (meters)')
        axes[1, 0].set_ylabel('Y (meters)')
        
        # Algorithm parameters
        axes[1, 1].text(0.1, 0.8, f"RANSAC Parameters:", transform=axes[1, 1].transAxes, fontsize=12, fontweight='bold')
        axes[1, 1].text(0.1, 0.7, f"Distance Threshold: {segmenter.ransac_params['distance_threshold']} m", transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f"Iterations: {segmenter.ransac_params['num_iterations']}", transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.5, f"Min Points: {segmenter.ransac_params['ransac_n']}", transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.3, f"Results:", transform=axes[1, 1].transAxes, fontsize=12, fontweight='bold')
        axes[1, 1].text(0.1, 0.2, f"Ground Ratio: {analysis['ground_ratio']*100:.1f}%", transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.1, f"Plane Angle: {analysis['angle_with_horizontal']:.1f}°", transform=axes[1, 1].transAxes)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        summary_path = os.path.join(output_dir, f"ground_segmentation_analysis_frame_{frame_idx:06d}.png")
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved analysis summary to: {summary_path}")
        
        # Show interactive visualization
        show_interactive = input("\nShow interactive 3D visualization? (y/n): ").lower().strip()
        if show_interactive == 'y':
            segmenter.visualize_interactive(vis_pcd)
        
        print("\nGround plane segmentation completed successfully!")
        print("\nGenerated files:")
        print(f"  - {side_view_path}")
        print(f"  - {top_view_path}")
        print(f"  - {front_view_path}")
        print(f"  - {summary_path}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the KITTI dataset is properly downloaded and the path is correct.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()