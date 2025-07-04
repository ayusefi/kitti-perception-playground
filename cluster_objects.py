#!/usr/bin/env python3
"""
KITTI Object Clustering Script using DBSCAN with Bounding Box Filtering

This script performs object clustering on non-ground points from LiDAR data using
a structured PerceptionPipeline class:
1. Loads point cloud and removes ground plane using RANSAC
2. Clusters non-ground points using DBSCAN
3. Filters clusters based on bounding box volume and point count
4. Saves top view cluster visualization as a PNG file
5. Opens interactive visualization for user interaction

Dependencies:
- numpy
- open3d

Usage:
    python cluster_objects.py
"""

import numpy as np
import open3d as o3d
import os

class PerceptionPipeline:
    """
    Complete perception pipeline for LiDAR object detection and clustering
    
    This class encapsulates all functionality for processing LiDAR point clouds:
    - Ground plane removal
    - Object clustering using DBSCAN
    - Cluster filtering based on size and point count
    - Visualization and output generation
    """
    
    def __init__(self, data_path: str, config: dict = None):
        """
        Initialize the perception pipeline
        
        Args:
            data_path: Path to KITTI dataset directory
            config: Optional configuration dictionary to override defaults
        """
        self.data_path = data_path
        self.config = self._load_config(config)
        
    def _load_config(self, config: dict = None) -> dict:
        """
        Load configuration with defaults and optional overrides
        
        Args:
            config: Optional configuration overrides
            
        Returns:
            Complete configuration dictionary
        """
        default_config = {
            'ransac': {
                'distance_threshold': 0.2,
                'ransac_n': 3,
                'num_iterations': 1000
            },
            'dbscan': {
                'eps': 0.7,          # Maximum distance between points in a cluster
                'min_points': 10     # Minimum points to form a cluster
            },
            'filter': {
                'min_volume': 1.0,   # Minimum bounding box volume in cubic meters
                'min_points': 20     # Minimum number of points in a cluster
            },
            'visualization': {
                'window_width': 1200,
                'window_height': 800,
                'zoom_level': 0.4
            }
        }
        
        if config:
            # Merge user config with defaults
            for section, params in config.items():
                if section in default_config:
                    default_config[section].update(params)
                else:
                    default_config[section] = params
                    
        return default_config

    def load_point_cloud(self, frame_idx: int) -> o3d.geometry.PointCloud:
        """
        Load a point cloud from KITTI dataset
        
        Args:
            frame_idx: Index of the frame to load
            
        Returns:
            Open3D PointCloud object
        """
        velodyne_path = os.path.join(self.data_path, "velodyne_points/data", f"{frame_idx:010d}.bin")
        if not os.path.exists(velodyne_path):
            raise FileNotFoundError(f"Point cloud file not found: {velodyne_path}")
        
        points = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)
        xyz_points = points[:, :3]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_points)
        return pcd

    def remove_ground(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Remove ground plane using RANSAC and return non-ground points
        
        Args:
            pcd: Input point cloud
            
        Returns:
            Point cloud containing only non-ground points
        """
        ransac_config = self.config['ransac']
        
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=ransac_config['distance_threshold'],
            ransac_n=ransac_config['ransac_n'],
            num_iterations=ransac_config['num_iterations']
        )
        
        all_indices = np.arange(len(pcd.points))
        object_indices = np.setdiff1d(all_indices, inliers)
        
        object_pcd = pcd.select_by_index(object_indices)
        print(f"Ground removal: Extracted {len(object_pcd.points)} non-ground points")
        return object_pcd

    def cluster_objects(self, pcd: o3d.geometry.PointCloud) -> list:
        """
        Cluster non-ground points using DBSCAN
        
        Args:
            pcd: Point cloud with non-ground points
            
        Returns:
            List of point cloud clusters
        """
        dbscan_config = self.config['dbscan']
        
        labels = np.array(pcd.cluster_dbscan(
            eps=dbscan_config['eps'],
            min_points=dbscan_config['min_points'],
            print_progress=True
        ))
        
        max_label = labels.max()
        print(f"DBSCAN clustering: Found {max_label + 1} clusters")
        
        clusters = []
        for i in range(max_label + 1):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) > 0:
                cluster = pcd.select_by_index(cluster_indices)
                clusters.append(cluster)
        
        # Handle noise points (label = -1)
        noise_indices = np.where(labels == -1)[0]
        if len(noise_indices) > 0:
            noise = pcd.select_by_index(noise_indices)
            clusters.append(noise)
        
        return clusters

    def filter_clusters(self, clusters: list) -> list:
        """
        Filter clusters based on bounding box volume and point count
        
        Args:
            clusters: List of point cloud clusters
            
        Returns:
            List of filtered clusters
        """
        filter_config = self.config['filter']
        filtered_clusters = []
        removed_count = 0
        
        print(f"\nFiltering clusters (min_volume: {filter_config['min_volume']} m³, min_points: {filter_config['min_points']})...")
        
        for i, cluster in enumerate(clusters):
            num_points = len(cluster.points)
            
            # Skip noise cluster (usually the last one with many points)
            if i == len(clusters) - 1 and num_points > 100:
                print(f"Cluster {i}: Noise cluster with {num_points} points - keeping")
                filtered_clusters.append(cluster)
                continue
            
            # Calculate bounding box volume
            obb = cluster.get_axis_aligned_bounding_box()
            extent = obb.get_extent()
            volume = extent[0] * extent[1] * extent[2]
            
            # Apply filters
            if volume >= filter_config['min_volume'] and num_points >= filter_config['min_points']:
                print(f"Cluster {i}: Volume={volume:.2f} m³, Points={num_points} - keeping")
                filtered_clusters.append(cluster)
            else:
                print(f"Cluster {i}: Volume={volume:.2f} m³, Points={num_points} - removing (too small)")
                removed_count += 1
        
        print(f"Filtering complete: Kept {len(filtered_clusters)} clusters, removed {removed_count} small clusters")
        return filtered_clusters

    def fit_bounding_boxes(self, clusters: list) -> list:
        """
        Fit axis-aligned bounding boxes to clusters
        
        Args:
            clusters: List of point cloud clusters
            
        Returns:
            List of bounding box geometries
        """
        bounding_boxes = []
        for cluster in clusters:
            obb = cluster.get_axis_aligned_bounding_box()
            obb.color = [0, 0, 1]  # Blue for all bounding boxes
            bounding_boxes.append(obb)
        
        print(f"Fitted {len(bounding_boxes)} bounding boxes")
        return bounding_boxes

    def visualize_results(self, clusters: list, bounding_boxes: list, output_path: str = None) -> None:
        """
        Visualize clusters with bounding boxes, save top view if output_path is provided,
        and open interactive visualization
        
        Args:
            clusters: List of point cloud clusters
            bounding_boxes: List of bounding box geometries
            output_path: Path to save the top view image (if provided)
        """
        vis_config = self.config['visualization']
        
        # Assign colors to clusters
        for i, cluster in enumerate(clusters):
            if i == len(clusters) - 1 and len(clusters) > 1:  # Noise points
                color = [0.5, 0.5, 0.5]  # Gray for noise
            else:
                color = np.random.random(3)  # Random color for objects
            cluster.colors = o3d.utility.Vector3dVector(np.tile(color, (len(cluster.points), 1)))
        
        # Save top view image if output_path is provided
        if output_path is not None:
            self._save_top_view(clusters, bounding_boxes, output_path, vis_config)
        
        # Open interactive visualization
        self._open_interactive_view(clusters, bounding_boxes, vis_config)

    def _save_top_view(self, clusters: list, bounding_boxes: list, output_path: str, vis_config: dict) -> None:
        """
        Save top view image of the clusters and bounding boxes
        
        Args:
            clusters: List of point cloud clusters
            bounding_boxes: List of bounding box geometries
            output_path: Path to save the image
            vis_config: Visualization configuration
        """
        vis_temp = o3d.visualization.Visualizer()
        vis_temp.create_window(width=vis_config['window_width'], height=vis_config['window_height'], visible=False)
        
        for cluster in clusters:
            vis_temp.add_geometry(cluster)
        for bbox in bounding_boxes:
            vis_temp.add_geometry(bbox)
            
        ctr_temp = vis_temp.get_view_control()
        ctr_temp.set_front([0, 0, -1])  # Top view
        ctr_temp.set_up([0, 1, 0])
        ctr_temp.set_lookat([0, 0, 0])
        ctr_temp.set_zoom(vis_config['zoom_level'])
        
        vis_temp.poll_events()
        vis_temp.update_renderer()
        vis_temp.capture_screen_image(output_path)
        vis_temp.destroy_window()
        
        print(f"Saved top view image to {output_path}")

    def _open_interactive_view(self, clusters: list, bounding_boxes: list, vis_config: dict) -> None:
        """
        Open interactive visualization window
        
        Args:
            clusters: List of point cloud clusters
            bounding_boxes: List of bounding box geometries
            vis_config: Visualization configuration
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="Perception Pipeline Results", 
            width=vis_config['window_width'], 
            height=vis_config['window_height']
        )
        
        for cluster in clusters:
            vis.add_geometry(cluster)
        for bbox in bounding_boxes:
            vis.add_geometry(bbox)
            
        ctr = vis.get_view_control()
        ctr.set_front([1, 0, 0])  # Side view
        ctr.set_up([0, 0, 1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_zoom(vis_config['zoom_level'])
        
        vis.run()
        vis.destroy_window()

    def run(self, frame_idx: int, output_dir: str = "output") -> None:
        """
        Execute the complete perception pipeline
        
        Args:
            frame_idx: Index of the frame to process
            output_dir: Directory to save output files
        """
        print("=" * 60)
        print("PERCEPTION PIPELINE EXECUTION")
        print("=" * 60)
        
        try:
            # Step 1: Load point cloud
            print(f"Step 1: Loading point cloud for frame {frame_idx}...")
            pcd = self.load_point_cloud(frame_idx)
            print(f"Loaded {len(pcd.points)} points")
            
            # Step 2: Remove ground plane
            print(f"\nStep 2: Removing ground plane...")
            object_pcd = self.remove_ground(pcd)
            
            # Step 3: Cluster objects
            print(f"\nStep 3: Clustering objects...")
            clusters = self.cluster_objects(object_pcd)
            
            # Step 4: Filter clusters
            print(f"\nStep 4: Filtering clusters...")
            filtered_clusters = self.filter_clusters(clusters)
            
            # Step 5: Fit bounding boxes
            print(f"\nStep 5: Fitting bounding boxes...")
            bounding_boxes = self.fit_bounding_boxes(filtered_clusters)
            
            # Step 6: Generate output
            print(f"\nStep 6: Generating visualization...")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"perception_pipeline_frame_{frame_idx:06d}.png")
            
            self.visualize_results(filtered_clusters, bounding_boxes, output_path)
            
            print("\n" + "=" * 60)
            print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
        except Exception as e:
            print(f"\nPipeline execution failed: {e}")
            raise

def main():
    """
    Main function demonstrating the PerceptionPipeline usage
    
    The main function is now clean and simple - it just creates a pipeline
    instance and calls the run() method.
    """
    # Configuration
    data_path = "data/2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync"
    frame_idx = 10
    output_dir = "output"
    
    # Optional: Custom configuration (uncomment to override defaults)
    # custom_config = {
    #     'filter': {
    #         'min_volume': 2.0,    # Require larger objects
    #         'min_points': 30      # Require more points
    #     },
    #     'dbscan': {
    #         'eps': 0.5,           # Tighter clustering
    #         'min_points': 15      # More points per cluster
    #     }
    # }
    
    # Create and run the perception pipeline
    pipeline = PerceptionPipeline(data_path)
    # pipeline = PerceptionPipeline(data_path, custom_config)  # With custom config
    pipeline.run(frame_idx, output_dir)

if __name__ == "__main__":
    main()