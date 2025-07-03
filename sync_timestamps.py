import os
import numpy as np
from datetime import datetime

def load_timestamps(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    dt_list = []
    for line in lines:
        s = line.strip()
        # Truncate nanoseconds to microseconds (6 digits) by slicing the string
        # Find dot position (start of fractional seconds)
        dot_pos = s.find('.')
        if dot_pos != -1:
            # Keep only first 6 digits after dot
            s = s[:dot_pos+7]  # dot + 6 digits
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")
        dt_list.append(dt)

    start_time = dt_list[0]
    elapsed_seconds = np.array([(dt - start_time).total_seconds() for dt in dt_list])
    return elapsed_seconds



def find_closest_index(target_timestamp, timestamps):
    """
    Given a target timestamp (float), find the index of closest timestamp in timestamps array.
    """
    index = np.argmin(np.abs(timestamps - target_timestamp))
    return index

class KittiTimestampSync:
    def __init__(self, lidar_ts_path, cam_ts_path):
        self.lidar_timestamps = load_timestamps(lidar_ts_path)
        self.camera_timestamps = load_timestamps(cam_ts_path)
        if len(self.lidar_timestamps) == 0 or len(self.camera_timestamps) == 0:
            raise ValueError("Timestamp files appear empty.")
    
    def get_closest_camera_idx(self, lidar_idx):
        """
        Given Lidar scan index, return closest camera frame index.
        """
        if lidar_idx < 0 or lidar_idx >= len(self.lidar_timestamps):
            raise IndexError("Lidar index out of bounds.")
        
        lidar_time = self.lidar_timestamps[lidar_idx]
        cam_idx = find_closest_index(lidar_time, self.camera_timestamps)
        return cam_idx

if __name__ == "__main__":
    # Adjust these paths to your KITTI sample folder
    base_folder = "data/2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync"
    lidar_ts_path = os.path.join(base_folder, "velodyne_points/timestamps.txt")
    cam_ts_path = os.path.join(base_folder, "image_00/timestamps.txt")

    sync = KittiTimestampSync(lidar_ts_path, cam_ts_path)

    # Example: find closest camera frame to Lidar index 10
    lidar_idx = 10
    cam_idx = sync.get_closest_camera_idx(lidar_idx)
    print(f"Lidar index {lidar_idx} timestamp: {sync.lidar_timestamps[lidar_idx]:.6f}")
    print(f"Closest camera index: {cam_idx} with timestamp {sync.camera_timestamps[cam_idx]:.6f}")
