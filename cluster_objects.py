#!/usr/bin/env python3
"""
KITTI Object Clustering Script using DBSCAN

This script performs object clustering on non-ground points from LiDAR data:
1. Loads point cloud and removes ground plane using RANSAC
2. Clusters non-ground points using DBSCAN
3. Saves top view cluster visualization as a PNG file
4. Opens interactive visualization for user interaction

Dependencies:
- numpy
- open3d

Usage:
    python cluster_objects.py
"""

import numpy as np
import open3d as o3d
import os

class ObjectClusterer:
    """Class to handle object clustering from point clouds"""
    
    def __init__(self, data_path: str):
        """
        Initialize the object clusterer
        
        Args:
            data_path: Path to KITTI dataset directory
        """
        self.data_path = data_path
        self.ransac_params = {
            'distance_threshold': 0.2,
            'ransac_n': 3,
            'num_iterations': 1000
        }
        self.dbscan_params = {
            'eps': 0.7,          # Maximum distance between points in a cluster
            'min_points': 10     # Minimum points to form a cluster
        }

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
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=self.ransac_params['distance_threshold'],
            ransac_n=self.ransac_params['ransac_n'],
            num_iterations=self.ransac_params['num_iterations']
        )
        
        all_indices = np.arange(len(pcd.points))
        object_indices = np.setdiff1d(all_indices, inliers)
        
        object_pcd = pcd.select_by_index(object_indices)
        print(f"Extracted {len(object_pcd.points)} non-ground points")
        return object_pcd

    def cluster_objects(self, pcd: o3d.geometry.PointCloud) -> list:
        """
        Cluster non-ground points using DBSCAN
        
        Args:
            pcd: Point cloud with non-ground points
            
        Returns:
            List of point cloud clusters
        """
        labels = np.array(pcd.cluster_dbscan(
            eps=self.dbscan_params['eps'],
            min_points=self.dbscan_params['min_points'],
            print_progress=True
        ))
        
        max_label = labels.max()
        print(f"Found {max_label + 1} clusters")
        
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

    def visualize_clusters(self, clusters: list, output_path: str = None) -> None:
        """
        Visualize clusters with random colors and their OBBs in blue, save top view if output_path is provided,
        and open an interactive visualization.

        Args:
            clusters: List of point cloud clusters
            output_path: Path to save the top view image (if provided)
        """
        # Assign colors to clusters and create OBBs
        obbs = []
        for i, cluster in enumerate(clusters):
            if i == len(clusters) - 1 and len(clusters) > 1:  # Noise points
                color = [0.5, 0.5, 0.5]  # Gray for noise
            else:
                color = np.random.random(3)  # Random color for objects
            cluster.colors = o3d.utility.Vector3dVector(np.tile(color, (len(cluster.points), 1)))
            obb = cluster.get_axis_aligned_bounding_box()
            obb.color = [0, 0, 1]  # Blue for all OBBs
            obbs.append(obb)
        
        # Save top view image if output_path is provided
        if output_path is not None:
            vis_temp = o3d.visualization.Visualizer()
            vis_temp.create_window(width=1200, height=800, visible=False)
            for cluster in clusters:
                vis_temp.add_geometry(cluster)
            for obb in obbs:
                vis_temp.add_geometry(obb)
            ctr_temp = vis_temp.get_view_control()
            ctr_temp.set_front([0, 0, -1])  # Top view
            ctr_temp.set_up([0, -1, 0])
            ctr_temp.set_lookat([0, 0, 0])
            ctr_temp.set_zoom(0.3)
            vis_temp.poll_events()
            vis_temp.update_renderer()
            vis_temp.capture_screen_image(output_path)
            vis_temp.destroy_window()
            print(f"Saved top view image to {output_path}")
        
        # Open interactive visualization starting from side view
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Object Clusters with OBBs", width=1200, height=800)
        for cluster in clusters:
            vis.add_geometry(cluster)
        for obb in obbs:
            vis.add_geometry(obb)
        ctr = vis.get_view_control()
        ctr.set_front([1, 0, 0])  # Side view
        ctr.set_up([0, 0, 1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_zoom(0.3)
        vis.run()
        vis.destroy_window()

def main():
    """Main function to demonstrate object clustering, save top view, and show interactive visualization"""
    # Adjust this path to your KITTI dataset directory
    data_path = "data/2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync"
    frame_idx = 10
    output_dir = "output"
    
    try:
        print("Initializing Object Clusterer...")
        clusterer = ObjectClusterer(data_path)
        
        print(f"Loading point cloud for frame {frame_idx}...")
        pcd = clusterer.load_point_cloud(frame_idx)
        print(f"Loaded {len(pcd.points)} points")
        
        print("Removing ground plane...")
        object_pcd = clusterer.remove_ground(pcd)
        
        print(" Henning objects with DBSCAN...")
        clusters = clusterer.cluster_objects(object_pcd)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Construct output filename
        output_path = os.path.join(output_dir, f"cluster_objects_frame_{frame_idx:06d}.png")
        
        print("Visualizing clusters...")
        clusterer.visualize_clusters(clusters, output_path=output_path)
        
        print("Object clustering completed successfully!")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()