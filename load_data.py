# load_data.py

import numpy as np
import open3d as o3d
import os

def load_kitti_bin_point_cloud(bin_path):
    """Load a .bin Lidar file from KITTI into a Nx4 NumPy array (x, y, z, reflectance)."""
    point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return point_cloud

def visualize_point_cloud(pc_np):
    """Convert numpy array to Open3D point cloud and visualize."""
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(pc_np[:, :3])  # only use x, y, z
    o3d.visualization.draw_geometries([pc_o3d])

if __name__ == "__main__":
    # Adjust this path to your local copy of KITTI raw data
    bin_file = "data/2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/0000000001.bin"
    
    if not os.path.exists(bin_file):
        raise FileNotFoundError(f"Cannot find file at {bin_file}")
    
    pc = load_kitti_bin_point_cloud(bin_file)
    visualize_point_cloud(pc)
