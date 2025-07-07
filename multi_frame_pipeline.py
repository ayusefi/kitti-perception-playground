#!/usr/bin/env python3
"""
Multi-Frame Perception Pipeline for Object Tracking

This module extends the single-frame PerceptionPipeline to handle sequences
of frames, providing the foundation for object tracking over time.

Dependencies:
- numpy
- open3d
- The original cluster_objects.py module
"""

import numpy as np
import open3d as o3d
import os
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from cluster_objects import PerceptionPipeline


@dataclass
class Detection:
    """
    Represents a single object detection in a frame
    """
    id: int
    frame_idx: int
    timestamp: float
    center: np.ndarray  # [x, y, z]
    dimensions: np.ndarray  # [length, width, height]
    points: np.ndarray  # Original point cloud points
    confidence: float = 1.0
    
    def __post_init__(self):
        """Ensure arrays are numpy arrays"""
        if not isinstance(self.center, np.ndarray):
            self.center = np.array(self.center)
        if not isinstance(self.dimensions, np.ndarray):
            self.dimensions = np.array(self.dimensions)
        if not isinstance(self.points, np.ndarray):
            self.points = np.array(self.points)


@dataclass
class FrameData:
    """
    Container for all detections in a single frame
    """
    frame_idx: int
    timestamp: float
    detections: List[Detection]
    total_points: int
    processing_time: float


class MultiFramePerceptionPipeline:
    """
    Multi-frame perception pipeline that processes sequences of LiDAR frames
    and extracts object detections for tracking
    """
    
    def __init__(self, data_path: str, config: dict = None):
        """
        Initialize the multi-frame pipeline
        
        Args:
            data_path: Path to KITTI dataset directory
            config: Optional configuration dictionary
        """
        self.data_path = data_path
        self.config = config or {}
        
        # Initialize single-frame pipeline
        self.single_frame_pipeline = PerceptionPipeline(data_path, config)
        
        # Storage for frame data
        self.frame_data: List[FrameData] = []
        self.detection_id_counter = 0
        
        # Performance tracking
        self.total_processing_time = 0.0
        
    def get_available_frames(self) -> List[int]:
        """
        Get list of available frame indices in the dataset
        
        Returns:
            List of frame indices
        """
        velodyne_path = os.path.join(self.data_path, "velodyne_points/data")
        if not os.path.exists(velodyne_path):
            raise FileNotFoundError(f"Velodyne data directory not found: {velodyne_path}")
        
        frame_files = [f for f in os.listdir(velodyne_path) if f.endswith('.bin')]
        frame_indices = [int(f.replace('.bin', '')) for f in frame_files]
        return sorted(frame_indices)
    
    def process_single_frame(self, frame_idx: int) -> FrameData:
        """
        Process a single frame and extract detections
        
        Args:
            frame_idx: Index of the frame to process
            
        Returns:
            FrameData object containing all detections
        """
        start_time = time.time()
        
        print(f"Processing frame {frame_idx}...")
        
        # Load point cloud
        pcd = self.single_frame_pipeline.load_point_cloud(frame_idx)
        total_points = len(pcd.points)
        
        # Remove ground
        object_pcd = self.single_frame_pipeline.remove_ground(pcd)
        
        # Cluster objects
        clusters = self.single_frame_pipeline.cluster_objects(object_pcd)
        
        # Filter clusters
        filtered_clusters = self.single_frame_pipeline.filter_clusters(clusters)
        
        # Convert clusters to detections
        detections = self._clusters_to_detections(filtered_clusters, frame_idx)
        
        processing_time = time.time() - start_time
        self.total_processing_time += processing_time
        
        # Create frame data
        frame_data = FrameData(
            frame_idx=frame_idx,
            timestamp=frame_idx * 0.1,  # Assuming 10 Hz LiDAR
            detections=detections,
            total_points=total_points,
            processing_time=processing_time
        )
        
        print(f"Frame {frame_idx}: Found {len(detections)} objects in {processing_time:.3f}s")
        
        return frame_data
    
    def _clusters_to_detections(self, clusters: List[o3d.geometry.PointCloud], frame_idx: int) -> List[Detection]:
        """
        Convert point cloud clusters to Detection objects
        
        Args:
            clusters: List of point cloud clusters
            frame_idx: Frame index
            
        Returns:
            List of Detection objects
        """
        detections = []
        
        for cluster in clusters:
            if len(cluster.points) == 0:
                continue
            
            # Get cluster points
            points = np.asarray(cluster.points)
            
            # Calculate bounding box
            bbox = cluster.get_axis_aligned_bounding_box()
            center = bbox.get_center()
            dimensions = bbox.get_extent()
            
            # Calculate confidence based on number of points and volume
            volume = dimensions[0] * dimensions[1] * dimensions[2]
            confidence = min(1.0, len(points) / 50.0)  # Normalize by expected points
            
            # Create detection
            detection = Detection(
                id=self.detection_id_counter,
                frame_idx=frame_idx,
                timestamp=frame_idx * 0.1,
                center=center,
                dimensions=dimensions,
                points=points,
                confidence=confidence
            )
            
            detections.append(detection)
            self.detection_id_counter += 1
        
        return detections
    
    def process_frame_sequence(self, start_frame: int, end_frame: int, 
                             step: int = 1, save_results: bool = True) -> List[FrameData]:
        """
        Process a sequence of frames
        
        Args:
            start_frame: Starting frame index
            end_frame: Ending frame index (inclusive)
            step: Frame step size
            save_results: Whether to save results to self.frame_data
            
        Returns:
            List of FrameData objects
        """
        print("=" * 80)
        print(f"MULTI-FRAME PROCESSING: Frames {start_frame} to {end_frame} (step={step})")
        print("=" * 80)
        
        available_frames = self.get_available_frames()
        frames_to_process = range(start_frame, end_frame + 1, step)
        
        # Filter to only available frames
        valid_frames = [f for f in frames_to_process if f in available_frames]
        
        if not valid_frames:
            raise ValueError(f"No valid frames found in range {start_frame}-{end_frame}")
        
        print(f"Processing {len(valid_frames)} frames: {valid_frames}")
        
        sequence_data = []
        
        for frame_idx in valid_frames:
            try:
                frame_data = self.process_single_frame(frame_idx)
                sequence_data.append(frame_data)
                
                if save_results:
                    self.frame_data.append(frame_data)
                    
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                continue
        
        avg_processing_time = np.mean([fd.processing_time for fd in sequence_data])
        total_detections = sum(len(fd.detections) for fd in sequence_data)
        
        print("\n" + "=" * 80)
        print("SEQUENCE PROCESSING SUMMARY")
        print("=" * 80)
        print(f"Frames processed: {len(sequence_data)}")
        print(f"Total detections: {total_detections}")
        print(f"Average processing time: {avg_processing_time:.3f}s per frame")
        print(f"Total processing time: {self.total_processing_time:.3f}s")
        
        return sequence_data
    
    def get_detections_by_frame(self, frame_idx: int) -> List[Detection]:
        """
        Get all detections for a specific frame
        
        Args:
            frame_idx: Frame index
            
        Returns:
            List of detections for the frame
        """
        for frame_data in self.frame_data:
            if frame_data.frame_idx == frame_idx:
                return frame_data.detections
        return []
    
    def get_detection_centers_by_frame(self, frame_idx: int) -> np.ndarray:
        """
        Get detection centers for a specific frame
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Array of detection centers [N, 3]
        """
        detections = self.get_detections_by_frame(frame_idx)
        if not detections:
            return np.empty((0, 3))
        
        centers = np.array([det.center for det in detections])
        return centers
    
    def save_sequence_data(self, output_path: str) -> None:
        """
        Save sequence data to file
        
        Args:
            output_path: Path to save the data
        """
        import pickle
        
        # Prepare data for saving (exclude large point clouds)
        save_data = []
        for frame_data in self.frame_data:
            frame_dict = {
                'frame_idx': frame_data.frame_idx,
                'timestamp': frame_data.timestamp,
                'total_points': frame_data.total_points,
                'processing_time': frame_data.processing_time,
                'detections': []
            }
            
            for det in frame_data.detections:
                det_dict = {
                    'id': det.id,
                    'frame_idx': det.frame_idx,
                    'timestamp': det.timestamp,
                    'center': det.center,
                    'dimensions': det.dimensions,
                    'confidence': det.confidence,
                    'num_points': len(det.points)
                }
                frame_dict['detections'].append(det_dict)
            
            save_data.append(frame_dict)
        
        with open(output_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Saved sequence data to {output_path}")
    
    def load_sequence_data(self, input_path: str) -> None:
        """
        Load sequence data from file
        
        Args:
            input_path: Path to load the data from
        """
        import pickle
        
        with open(input_path, 'rb') as f:
            save_data = pickle.load(f)
        
        # Convert back to frame data (without point clouds)
        self.frame_data = []
        for frame_dict in save_data:
            detections = []
            for det_dict in frame_dict['detections']:
                # Create detection without points
                detection = Detection(
                    id=det_dict['id'],
                    frame_idx=det_dict['frame_idx'],
                    timestamp=det_dict['timestamp'],
                    center=det_dict['center'],
                    dimensions=det_dict['dimensions'],
                    points=np.empty((0, 3)),  # Empty points array
                    confidence=det_dict['confidence']
                )
                detections.append(detection)
            
            frame_data = FrameData(
                frame_idx=frame_dict['frame_idx'],
                timestamp=frame_dict['timestamp'],
                detections=detections,
                total_points=frame_dict['total_points'],
                processing_time=frame_dict['processing_time']
            )
            self.frame_data.append(frame_data)
        
        print(f"Loaded sequence data from {input_path}")
    
    def print_statistics(self) -> None:
        """Print statistics about the processed sequence"""
        if not self.frame_data:
            print("No frame data available")
            return
        
        print("\n" + "=" * 60)
        print("SEQUENCE STATISTICS")
        print("=" * 60)
        
        total_frames = len(self.frame_data)
        total_detections = sum(len(fd.detections) for fd in self.frame_data)
        avg_detections = total_detections / total_frames if total_frames > 0 else 0
        
        processing_times = [fd.processing_time for fd in self.frame_data]
        avg_processing_time = np.mean(processing_times)
        
        print(f"Total frames processed: {total_frames}")
        print(f"Total detections: {total_detections}")
        print(f"Average detections per frame: {avg_detections:.2f}")
        print(f"Average processing time: {avg_processing_time:.3f}s")
        print(f"Total processing time: {self.total_processing_time:.3f}s")
        
        # Frame-by-frame summary
        print("\nFrame-by-frame summary:")
        for fd in self.frame_data:
            print(f"Frame {fd.frame_idx:3d}: {len(fd.detections):2d} objects, "
                  f"{fd.processing_time:.3f}s, {fd.total_points:5d} points")


def main():
    """
    Main function demonstrating multi-frame processing
    """
    # Configuration
    data_path = "data/2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync"
    
    # Custom configuration for better object detection
    config = {
        'filter': {
            'min_volume': 0.5,  # Smaller minimum volume
            'min_points': 15    # Fewer points required
        },
        'dbscan': {
            'eps': 0.8,         # Slightly looser clustering
            'min_points': 8     # Fewer points per cluster
        }
    }
    
    # Create multi-frame pipeline
    pipeline = MultiFramePerceptionPipeline(data_path, config)
    
    # Process a sequence of frames
    start_frame = 10
    end_frame = 20
    
    try:
        # Process frames
        sequence_data = pipeline.process_frame_sequence(start_frame, end_frame)
        
        # Print statistics
        pipeline.print_statistics()
        
        # Save results
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save sequence data
        pipeline.save_sequence_data(os.path.join(output_dir, "sequence_data.pkl"))
        
        # Print detection centers for tracking preparation
        print("\nDetection centers per frame (for tracking):")
        for frame_data in sequence_data:
            centers = pipeline.get_detection_centers_by_frame(frame_data.frame_idx)
            print(f"Frame {frame_data.frame_idx}: {len(centers)} detections")
            for i, center in enumerate(centers):
                print(f"  Object {i}: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
        
    except Exception as e:
        print(f"Error in multi-frame processing: {e}")
        raise


if __name__ == "__main__":
    main()