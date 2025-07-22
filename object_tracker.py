#!/usr/bin/env python3
"""
Multi-Object Tracker using Kalman Filter and Hungarian Algorithm Association

This module implements a complete multi-object tracker that:
1. Uses Kalman filters to track individual objects
2. Performs optimal data association using the Hungarian algorithm
3. Handles track initialization and termination
4. Provides visualization capabilities

Dependencies:
- numpy
- scipy (for linear_sum_assignment)
- kalman_filter module
- multi_frame_pipeline module
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from scipy.optimize import linear_sum_assignment

# Import our custom modules
from kalman_filter import KalmanFilter
from multi_frame_pipeline import MultiFramePerceptionPipeline, Detection, FrameData


class TrackState(Enum):
    """Track state enumeration"""
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    DELETED = "deleted"


@dataclass
class Track:
    """
    Represents a single object track
    """
    id: int
    kalman_filter: KalmanFilter
    state: TrackState
    hits: int  # Number of successful updates
    time_since_update: int  # Frames since last update
    age: int  # Total age of the track
    last_detection: Optional[Detection] = None
    
    def __post_init__(self):
        self.creation_time = time.time()
    
    def get_position(self) -> np.ndarray:
        """Get current position estimate"""
        return self.kalman_filter.get_position()
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate"""
        return self.kalman_filter.get_velocity()
    
    def predict(self) -> None:
        """Predict next state"""
        self.kalman_filter.predict()
        self.time_since_update += 1
        self.age += 1
    
    def update(self, detection: Detection) -> None:
        """Update with detection"""
        self.kalman_filter.update(detection.center)
        self.hits += 1
        self.time_since_update = 0
        self.last_detection = detection
        
        # Update track state
        if self.state == TrackState.TENTATIVE and self.hits >= 3:
            self.state = TrackState.CONFIRMED
    
    def mark_missed(self) -> None:
        """Mark track as missed (no detection associated)"""
        self.time_since_update += 1
        self.age += 1
        
        # Mark for deletion if missed for too long
        if self.time_since_update > 5:
            self.state = TrackState.DELETED


class MultiObjectTracker:
    """
    Multi-Object Tracker using Kalman Filter and Hungarian Algorithm Association
    
    This class manages multiple object tracks and performs optimal data association
    between detections and existing tracks using the Hungarian algorithm.
    
    The Hungarian algorithm solves the assignment problem optimally, ensuring
    the minimum total cost assignment between tracks and detections.
    """
    
    def __init__(self, max_distance: float = 5.0, dt: float = 0.1):
        """
        Initialize the tracker
        
        Args:
            max_distance: Maximum distance for data association (meters)
            dt: Time step between frames (seconds)
        """
        self.max_distance = max_distance
        self.dt = dt
        self.tracks: List[Track] = []
        self.track_id_counter = 0
        self.frame_count = 0
        
        # Tracking history for visualization
        self.track_history: Dict[int, List[np.ndarray]] = {}
        
        # Performance metrics
        self.processing_times = []
        
    def predict(self) -> None:
        """Predict all tracks to current frame"""
        for track in self.tracks:
            track.predict()
    
    def _calculate_cost_matrix(self, detections: List[Detection]) -> np.ndarray:
        """
        Calculate cost matrix for data association
        
        Args:
            detections: List of detections
            
        Returns:
            Cost matrix [tracks x detections]
        """
        if not self.tracks or not detections:
            return np.array([])
        
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for i, track in enumerate(self.tracks):
            track_pos = track.get_position()
            for j, detection in enumerate(detections):
                # Use Euclidean distance as cost
                distance = np.linalg.norm(track_pos - detection.center)
                cost_matrix[i, j] = distance
        
        return cost_matrix
    
    def _associate_detections_to_tracks(self, detections: List[Detection]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections to tracks using the Hungarian algorithm (optimal assignment)
        
        Args:
            detections: List of detections
            
        Returns:
            Tuple of (matches, unmatched_detections, unmatched_tracks)
        """
        if not self.tracks or not detections:
            return [], list(range(len(detections))), list(range(len(self.tracks)))
        
        cost_matrix = self._calculate_cost_matrix(detections)
        
        # Use Hungarian algorithm for optimal assignment
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))
        
        # Process the optimal assignments
        for track_idx, det_idx in zip(track_indices, detection_indices):
            # Only accept assignments below the distance threshold
            if cost_matrix[track_idx, det_idx] < self.max_distance:
                matches.append((track_idx, det_idx))
                if det_idx in unmatched_detections:
                    unmatched_detections.remove(det_idx)
                if track_idx in unmatched_tracks:
                    unmatched_tracks.remove(track_idx)
        
        return matches, unmatched_detections, unmatched_tracks
    
    def _initiate_new_tracks(self, detections: List[Detection], unmatched_detections: List[int]) -> None:
        """
        Create new tracks for unmatched detections
        
        Args:
            detections: List of all detections
            unmatched_detections: Indices of unmatched detections
        """
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            
            # Create new Kalman filter
            kf = KalmanFilter(dt=self.dt)
            kf.initialize(detection.center)
            
            # Create new track
            track = Track(
                id=self.track_id_counter,
                kalman_filter=kf,
                state=TrackState.TENTATIVE,
                hits=1,
                time_since_update=0,
                age=1,
                last_detection=detection
            )
            
            self.tracks.append(track)
            self.track_history[self.track_id_counter] = [detection.center.copy()]
            self.track_id_counter += 1
    
    def _update_tracks(self, detections: List[Detection], matches: List[Tuple[int, int]]) -> None:
        """
        Update matched tracks with detections
        
        Args:
            detections: List of detections
            matches: List of (track_idx, detection_idx) pairs
        """
        for track_idx, det_idx in matches:
            detection = detections[det_idx]
            track = self.tracks[track_idx]
            
            # Update track
            track.update(detection)
            
            # Update history
            if track.id not in self.track_history:
                self.track_history[track.id] = []
            self.track_history[track.id].append(detection.center.copy())
    
    def _mark_missed_tracks(self, unmatched_tracks: List[int]) -> None:
        """
        Mark unmatched tracks as missed
        
        Args:
            unmatched_tracks: Indices of unmatched tracks
        """
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
    
    def _delete_old_tracks(self) -> None:
        """Remove tracks marked for deletion"""
        self.tracks = [track for track in self.tracks if track.state != TrackState.DELETED]
    
    def update(self, detections: List[Detection], verbose: bool = False) -> None:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detections for current frame
            verbose: Whether to print cost matrix details for debugging
        """
        start_time = time.time()
        
        # Step 1: Predict all tracks
        self.predict()
        
        # Optional: Print cost matrix for debugging
        if verbose:
            self.print_cost_matrix(detections, verbose=True)
        
        # Step 2: Data association using Hungarian algorithm
        matches, unmatched_detections, unmatched_tracks = self._associate_detections_to_tracks(detections)
        
        # Step 3: Update matched tracks
        self._update_tracks(detections, matches)
        
        # Step 4: Mark missed tracks
        self._mark_missed_tracks(unmatched_tracks)
        
        # Step 5: Create new tracks
        self._initiate_new_tracks(detections, unmatched_detections)
        
        # Step 6: Delete old tracks
        self._delete_old_tracks()
        
        self.frame_count += 1
        self.processing_times.append(time.time() - start_time)
    
    def get_active_tracks(self) -> List[Track]:
        """Get all active (confirmed) tracks"""
        return [track for track in self.tracks if track.state == TrackState.CONFIRMED]
    
    def get_all_tracks(self) -> List[Track]:
        """Get all tracks (including tentative)"""
        return self.tracks
    
    def get_track_positions(self) -> Dict[int, np.ndarray]:
        """Get current positions of all tracks"""
        positions = {}
        for track in self.tracks:
            positions[track.id] = track.get_position()
        return positions
    
    def get_track_history(self, track_id: int) -> List[np.ndarray]:
        """Get position history for a specific track"""
        return self.track_history.get(track_id, [])
    
    def print_statistics(self) -> None:
        """Print tracking statistics"""
        active_tracks = self.get_active_tracks()
        
        print(f"\n{'='*60}")
        print("TRACKING STATISTICS")
        print(f"{'='*60}")
        print(f"Frame: {self.frame_count}")
        print(f"Total tracks: {len(self.tracks)}")
        print(f"Active tracks: {len(active_tracks)}")
        print(f"Track ID counter: {self.track_id_counter}")
        
        if self.processing_times:
            avg_time = np.mean(self.processing_times)
            print(f"Average processing time: {avg_time:.4f}s")
        
        print(f"\nTrack details:")
        for track in self.tracks:
            pos = track.get_position()
            vel = track.get_velocity()
            speed = np.linalg.norm(vel)
            print(f"  Track {track.id}: {track.state.value}, "
                  f"pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], "
                  f"speed={speed:.2f}, hits={track.hits}, age={track.age}")

    def print_cost_matrix(self, detections: List[Detection], verbose: bool = False) -> None:
        """
        Print the cost matrix for debugging Hungarian algorithm
        
        Args:
            detections: List of current detections
            verbose: Whether to print detailed cost matrix
        """
        if not self.tracks or not detections or not verbose:
            return
            
        cost_matrix = self._calculate_cost_matrix(detections)
        
        print(f"\n{'='*50}")
        print("COST MATRIX (Hungarian Algorithm Input)")
        print(f"{'='*50}")
        print(f"Tracks: {len(self.tracks)}, Detections: {len(detections)}")
        
        # Header
        header = "      "
        for j in range(len(detections)):
            header += f"  Det{j:2d} "
        print(header)
        
        # Matrix rows
        for i in range(len(self.tracks)):
            row = f"Trk{i:2d} "
            for j in range(len(detections)):
                row += f"{cost_matrix[i, j]:6.2f} "
            print(row)
        
        # Show Hungarian assignment
        from scipy.optimize import linear_sum_assignment
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)
        
        print(f"\nHungarian Algorithm Assignment:")
        total_cost = 0
        for track_idx, det_idx in zip(track_indices, detection_indices):
            cost = cost_matrix[track_idx, det_idx]
            if cost < self.max_distance:
                print(f"  Track {track_idx} -> Detection {det_idx} (cost: {cost:.3f}) ✓")
                total_cost += cost
            else:
                print(f"  Track {track_idx} -> Detection {det_idx} (cost: {cost:.3f}) ✗ (above threshold)")
        print(f"Total assignment cost: {total_cost:.3f}")


class TrackingVisualizer:
    """
    Visualizer for tracking results
    """
    
    def __init__(self, tracker: MultiObjectTracker):
        """
        Initialize visualizer
        
        Args:
            tracker: MultiObjectTracker instance
        """
        self.tracker = tracker
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.colors = plt.cm.Set3(np.linspace(0, 1, 20))  # 20 different colors
        
        # Fixed axis limits for consistent frame size
        self.x_min, self.x_max = -80, 80
        self.y_min, self.y_max = -30, 50
    
    def plot_frame(self, detections: List[Detection], frame_idx: int) -> None:
        """
        Plot current frame with tracks and detections
        
        Args:
            detections: Current detections
            frame_idx: Frame index
        """
        self.ax.clear()
        
        # Plot detections as red circles
        if detections:
            det_positions = np.array([det.center for det in detections])
            self.ax.scatter(det_positions[:, 0], det_positions[:, 1], 
                          c='red', s=100, alpha=0.7, label='Detections', marker='o')
            
            # Add detection IDs
            for i, det in enumerate(detections):
                self.ax.annotate(f'D{i}', (det.center[0], det.center[1]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8, color='red')
        
        # Plot tracks with different colors and trajectories
        for track in self.tracker.get_all_tracks():
            pos = track.get_position()
            track_color = self.colors[track.id % len(self.colors)]
            
            # Plot current position with different markers for track states
            if track.state == TrackState.CONFIRMED:
                marker = 'o'
                size = 150
                alpha = 1.0
            else:  # TENTATIVE
                marker = '^'
                size = 100
                alpha = 0.6
            
            self.ax.scatter(pos[0], pos[1], c=[track_color], s=size, 
                          marker=marker, alpha=alpha, edgecolors='black', linewidth=1)
            
            # Add track ID labels
            self.ax.annotate(f'T{track.id}', (pos[0], pos[1]), 
                           xytext=(5, -15), textcoords='offset points', 
                           fontsize=10, fontweight='bold', color='black',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=track_color, alpha=0.7))
            
            # Plot trajectory history
            history = self.tracker.get_track_history(track.id)
            if len(history) > 1:
                history_array = np.array(history)
                self.ax.plot(history_array[:, 0], history_array[:, 1], 
                           c=track_color, alpha=0.8, linewidth=2, linestyle='-')
                
                # Add dots for historical positions
                self.ax.scatter(history_array[:-1, 0], history_array[:-1, 1], 
                              c=track_color, s=20, alpha=0.4)
            
            # Plot velocity vector as arrow (shorter arrows)
            vel = track.get_velocity()
            if np.linalg.norm(vel) > 0.5:  # Only show significant velocities
                # Scale down the arrow length significantly
                arrow_scale = 0.3  # Much shorter arrows
                self.ax.arrow(pos[0], pos[1], vel[0] * arrow_scale, vel[1] * arrow_scale, 
                            head_width=0.8, head_length=0.6, 
                            fc=track_color, ec=track_color, alpha=0.8, linewidth=1.5)
        
        # Customize plot
        self.ax.set_xlabel('X Position (meters)', fontsize=12)
        self.ax.set_ylabel('Y Position (meters)', fontsize=12)
        self.ax.set_title(f'Multi-Object Tracking - Frame {frame_idx}\n'
                         f'Active Tracks: {len(self.tracker.get_active_tracks())}, '
                         f'Detections: {len(detections)}', fontsize=14)
        self.ax.grid(True, alpha=0.3)
        
        # Use fixed axis limits for consistent frame size
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        
        # Legend
        legend_elements = []
        if detections:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor='red', markersize=10, 
                                            label='Detections', alpha=0.7))
        if self.tracker.get_active_tracks():
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor='blue', markersize=12, 
                                            label='Confirmed Tracks'))
        tentative_tracks = [t for t in self.tracker.tracks if t.state == TrackState.TENTATIVE]
        if tentative_tracks:
            legend_elements.append(plt.Line2D([0], [0], marker='^', color='w', 
                                            markerfacecolor='gray', markersize=10, 
                                            label='Tentative Tracks', alpha=0.6))
        
        if legend_elements:
            self.ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
    
    def save_frame(self, detections: List[Detection], frame_idx: int, output_path: str) -> None:
        """
        Save current frame plot
        
        Args:
            detections: Current detections
            frame_idx: Frame index
            output_path: Path to save the plot
        """
        self.plot_frame(detections, frame_idx)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved tracking visualization to {output_path}")
    
    def create_gif(self, frame_paths: List[str], output_path: str, duration: float = 0.5) -> None:
        """
        Create a GIF from saved frame images
        
        Args:
            frame_paths: List of paths to frame images
            output_path: Path to save the GIF
            duration: Duration of each frame in seconds
        """
        try:
            from PIL import Image
            
            # Load all images
            images = []
            for path in frame_paths:
                if os.path.exists(path):
                    img = Image.open(path)
                    images.append(img)
            
            if images:
                # Save as GIF
                images[0].save(
                    output_path,
                    save_all=True,
                    append_images=images[1:],
                    duration=int(duration * 1000),  # Convert to milliseconds
                    loop=0
                )
                print(f"✅ Created tracking GIF: {output_path}")
                print(f"   📊 {len(images)} frames, {duration}s per frame")
            else:
                print("❌ No frame images found for GIF creation")
                
        except ImportError:
            print("❌ PIL (Pillow) not available. Install with: pip install Pillow")
        except Exception as e:
            print(f"❌ Error creating GIF: {e}")


def main():
    """
    Main function demonstrating the complete tracking pipeline
    """
    print("=" * 80)
    print("MULTI-OBJECT TRACKING DEMONSTRATION")
    print("=" * 80)
    
    # Configuration
    data_path = "data/2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync"
    
    # Perception pipeline config
    perception_config = {
        'filter': {
            'min_volume': 0.5,
            'min_points': 15
        },
        'dbscan': {
            'eps': 0.8,
            'min_points': 8
        }
    }
    
    # Create pipeline and tracker
    pipeline = MultiFramePerceptionPipeline(data_path, perception_config)
    tracker = MultiObjectTracker(max_distance=3.0, dt=0.1)
    visualizer = TrackingVisualizer(tracker)
    
    # Process sequence
    start_frame = 0
    end_frame = 100
    
    try:
        print(f"Processing frames {start_frame} to {end_frame}...")
        
        # Create output directory
        output_dir = "output/tracking"
        os.makedirs(output_dir, exist_ok=True)
        
        # Track frame paths for GIF creation
        frame_paths = []
        
        # Process each frame
        for frame_idx in range(start_frame, end_frame + 1):
            print(f"\n--- Processing Frame {frame_idx} ---")
            
            # Get detections
            frame_data = pipeline.process_single_frame(frame_idx)
            detections = frame_data.detections
            
            # Update tracker
            tracker.update(detections)
            
            # Print statistics
            tracker.print_statistics()
            
            # Save visualization
            output_path = os.path.join(output_dir, f"tracking_frame_{frame_idx:06d}.png")
            visualizer.save_frame(detections, frame_idx, output_path)
            frame_paths.append(output_path)
        
        print("\n" + "=" * 80)
        print("TRACKING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        # Create GIF from all frames
        print(f"\n🎬 Creating tracking GIF...")
        gif_path = "output/tracking.gif"  # Direct path to tracking.gif
        visualizer.create_gif(frame_paths, gif_path, duration=0.2)
        
        # Final statistics
        print(f"\nFinal Statistics:")
        print(f"Total frames processed: {tracker.frame_count}")
        print(f"Total tracks created: {tracker.track_id_counter}")
        print(f"Active tracks: {len(tracker.get_active_tracks())}")
        
        # Track lifetimes
        print(f"\nTrack lifetimes:")
        for track in tracker.get_all_tracks():
            print(f"  Track {track.id}: {track.age} frames, {track.hits} hits")
        
        print(f"\n📺 View the tracking animation at: {gif_path}")
        
    except Exception as e:
        print(f"Error in tracking: {e}")
        raise


if __name__ == "__main__":
    main()