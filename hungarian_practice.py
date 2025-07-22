#!/usr/bin/env python3
"""
Hungarian Algorithm Practice Script

This script demonstrates the use of scipy.optimize.linear_sum_assignment
for solving the assignment problem in object tracking.

The Hungarian algorithm finds the optimal assignment that minimizes
the total cost when pairing two sets of items.

Dependencies:
- numpy
- scipy

Usage:
    python hungarian_practice.py
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import time


def greedy_assignment(cost_matrix):
    """
    Greedy assignment algorithm - assigns each track to nearest available detection
    """
    cost_matrix = np.array(cost_matrix)
    n_tracks, n_detections = cost_matrix.shape
    
    used_detections = set()
    total_cost = 0.0
    assignments = []
    
    # For each track, find the nearest available detection
    for track_idx in range(n_tracks):
        best_detection_idx = None
        best_cost = float('inf')
        
        # Find the cheapest available detection for this track
        for det_idx in range(n_detections):
            if det_idx not in used_detections:
                cost = cost_matrix[track_idx, det_idx]
                if cost < best_cost:
                    best_cost = cost
                    best_detection_idx = det_idx
        
        # Make the assignment if we found a detection
        if best_detection_idx is not None:
            used_detections.add(best_detection_idx)
            total_cost += best_cost
            assignments.append((track_idx, best_detection_idx))
    
    return total_cost


def hungarian_assignment(cost_matrix):
    """
    Hungarian assignment algorithm using scipy implementation
    """
    cost_matrix = np.array(cost_matrix)
    track_indices, detection_indices = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[track_indices, detection_indices].sum()
    return total_cost


def create_extreme_scenario():
    """
    Create a realistic 3D point cloud scenario with random object distribution
    and moving environment simulation - like real KITTI autonomous vehicle data
    """
    print("\n" + "=" * 70)
    print("REALISTIC 3D POINT CLOUD ASSIGNMENT - MOVING ENVIRONMENT")
    print("=" * 70)
    
    # Create realistic 3D point cloud with random distribution
    n = 12
    tracks = []
    detections = []
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Generate tracks with realistic 3D point cloud distribution
    print("Generating realistic 3D object positions...")
    for i in range(n):
        # Random 3D positions within sensor range (0-50m radius)
        # Simulate cars, pedestrians, cyclists at various distances and positions
        
        # Random distance from ego vehicle (5-45 meters)
        distance = np.random.uniform(5, 45)
        
        # Random angle around ego vehicle (full 360 degrees)
        angle = np.random.uniform(0, 2 * np.pi)
        
        # Convert to Cartesian coordinates with realistic spread
        x = distance * np.cos(angle) + np.random.normal(0, 2.0)  # Additional position noise
        y = distance * np.sin(angle) + np.random.normal(0, 2.0)
        
        # Realistic object heights (cars: 0.5-2.0m, pedestrians: 1.5-1.8m, etc.)
        if i % 3 == 0:  # Cars
            z = np.random.uniform(0.5, 2.0)
        elif i % 3 == 1:  # Pedestrians  
            z = np.random.uniform(1.5, 1.8)
        else:  # Cyclists, trucks, etc.
            z = np.random.uniform(0.8, 2.5)
        
        tracks.append([x, y, z])
    
    # Generate detections with movement simulation (objects have moved between frames)
    print("Simulating object movement and sensor measurements...")
    for i, track_pos in enumerate(tracks):
        # Simulate object movement between frames
        # Objects move with realistic velocities: cars (0-15 m/s), pedestrians (0-2 m/s)
        dt = 0.1  # 100ms between frames (10 Hz sensor)
        
        if i % 3 == 0:  # Cars - higher velocity
            velocity = np.random.uniform(0, 15)  # 0-54 km/h
        elif i % 3 == 1:  # Pedestrians - lower velocity
            velocity = np.random.uniform(0, 2)   # 0-7.2 km/h
        else:  # Cyclists, trucks
            velocity = np.random.uniform(0, 8)   # 0-29 km/h
        
        # Random movement direction
        movement_angle = np.random.uniform(0, 2 * np.pi)
        
        # Calculate new position after movement
        dx = velocity * dt * np.cos(movement_angle)
        dy = velocity * dt * np.sin(movement_angle)
        dz = np.random.normal(0, 0.05)  # Small vertical movement
        
        new_x = track_pos[0] + dx + np.random.normal(0, 0.3)  # Sensor noise
        new_y = track_pos[1] + dy + np.random.normal(0, 0.3)  # Sensor noise  
        new_z = track_pos[2] + dz + np.random.normal(0, 0.1)  # Height noise
        
        detections.append([new_x, new_y, new_z])
    
    tracks = np.array(tracks)
    detections = np.array(detections)
    
    # Calculate 3D Euclidean distances
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance = np.linalg.norm(tracks[i] - detections[j])
            cost_matrix[i, j] = distance
    
    print(f"\nCreated realistic {n}x{n} 3D point cloud with movement simulation")
    print("Track positions (previous frame, x,y,z in meters):")
    for i in range(min(6, n)):
        pos = tracks[i]
        obj_type = ["Car", "Pedestrian", "Cyclist/Truck"][i % 3]
        print(f"  Track {i} ({obj_type:10s}): [{pos[0]:6.1f}, {pos[1]:6.1f}, {pos[2]:4.1f}]")
    print("  ...")
    
    print("\nDetection positions (current frame, x,y,z in meters):")
    for i in range(min(6, n)):
        pos = detections[i]
        movement = np.linalg.norm(pos - tracks[i])
        obj_type = ["Car", "Pedestrian", "Cyclist/Truck"][i % 3]
        print(f"  Det   {i} ({obj_type:10s}): [{pos[0]:6.1f}, {pos[1]:6.1f}, {pos[2]:4.1f}] (moved: {movement:.2f}m)")
    print("  ...")
    
    print("\nCost matrix (3D distances between tracks and detections):")
    print("        ", end="")
    for j in range(min(8, n)):
        print(f"Det{j:2d}  ", end="")
    if n > 8:
        print("...")
    else:
        print()
    
    for i in range(min(8, n)):
        print(f"Track{i:2d}  ", end="")
        for j in range(min(8, n)):
            print(f"{cost_matrix[i, j]:5.2f} ", end="")
        if n > 8:
            print("...")
        else:
            print()
    if n > 8:
        print("  ⋮       ⋮     ⋮     ⋮")
    
    # Greedy assignment
    print(f"\nPerforming greedy assignment on {n}x{n} moving 3D objects...")
    greedy_start = time.time()
    
    greedy_cost = 0
    used = set()
    greedy_assignments = []
    
    for track_idx in range(n):
        min_cost = float('inf')
        best_det = -1
        for det_idx in range(n):
            if det_idx not in used and cost_matrix[track_idx, det_idx] < min_cost:
                min_cost = cost_matrix[track_idx, det_idx]
                best_det = det_idx
        if best_det != -1:
            used.add(best_det)
            greedy_cost += min_cost
            greedy_assignments.append((track_idx, best_det))
    
    greedy_time = time.time() - greedy_start
    
    # Hungarian assignment
    print(f"Performing Hungarian assignment on {n}x{n} moving 3D objects...")
    hungarian_start = time.time()
    
    track_indices, detection_indices = linear_sum_assignment(cost_matrix)
    hungarian_cost = cost_matrix[track_indices, detection_indices].sum()
    
    hungarian_time = time.time() - hungarian_start
    
    # Results
    improvement = greedy_cost - hungarian_cost
    percentage = (improvement / greedy_cost) * 100 if greedy_cost > 0 else 0
    
    print(f"\n🎯 REALISTIC 3D POINT CLOUD TRACKING RESULTS:")
    print(f"   Problem size:         {n}×{n} moving 3D objects")
    print(f"   Greedy total cost:    {greedy_cost:.3f}m")
    print(f"   Hungarian total cost: {hungarian_cost:.3f}m")
    print(f"   💰 Distance reduction: {improvement:.3f}m ({percentage:.1f}% improvement)")
    print(f"   ⏱️  Greedy time:       {greedy_time*1000:.2f} ms")
    print(f"   ⏱️  Hungarian time:    {hungarian_time*1000:.2f} ms")
    
    if improvement > 0.5:
        print(f"   🏆 Hungarian algorithm achieved SIGNIFICANT improvement!")
        print(f"   🚗 Better multi-object tracking with {improvement:.2f}m less total error")
        print(f"   📊 Reduced ID switches and improved trajectory continuity")
    elif improvement > 0.1:
        print(f"   ✅ Hungarian algorithm achieved measurable improvement!")
        print(f"   🚗 Better object association with {improvement:.2f}m less error")
    else:
        print(f"   ℹ️  Similar performance - both algorithms found near-optimal solutions")
    
    return greedy_cost, hungarian_cost, improvement, greedy_time, hungarian_time, cost_matrix


def get_extreme_scenario_data():
    """
    Get the cost matrix from the realistic 3D point cloud scenario for visualization
    """
    # Create realistic 3D point cloud with random distribution
    n = 12
    tracks = []
    detections = []
    
    # Set same random seed for consistent results
    np.random.seed(42)
    
    # Generate tracks with realistic 3D point cloud distribution
    for i in range(n):
        # Random 3D positions within sensor range
        distance = np.random.uniform(5, 45)
        angle = np.random.uniform(0, 2 * np.pi)
        
        x = distance * np.cos(angle) + np.random.normal(0, 2.0)
        y = distance * np.sin(angle) + np.random.normal(0, 2.0)
        
        if i % 3 == 0:  # Cars
            z = np.random.uniform(0.5, 2.0)
        elif i % 3 == 1:  # Pedestrians  
            z = np.random.uniform(1.5, 1.8)
        else:  # Cyclists, trucks, etc.
            z = np.random.uniform(0.8, 2.5)
        
        tracks.append([x, y, z])
    
    # Generate detections with movement simulation
    for i, track_pos in enumerate(tracks):
        dt = 0.1  # 100ms between frames
        
        if i % 3 == 0:  # Cars
            velocity = np.random.uniform(0, 15)
        elif i % 3 == 1:  # Pedestrians
            velocity = np.random.uniform(0, 2)
        else:  # Cyclists, trucks
            velocity = np.random.uniform(0, 8)
        
        movement_angle = np.random.uniform(0, 2 * np.pi)
        
        dx = velocity * dt * np.cos(movement_angle)
        dy = velocity * dt * np.sin(movement_angle)
        dz = np.random.normal(0, 0.05)
        
        new_x = track_pos[0] + dx + np.random.normal(0, 0.3)
        new_y = track_pos[1] + dy + np.random.normal(0, 0.3)
        new_z = track_pos[2] + dz + np.random.normal(0, 0.1)
        
        detections.append([new_x, new_y, new_z])
    
    tracks = np.array(tracks)
    detections = np.array(detections)
    
    # Calculate 3D Euclidean distances
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance = np.linalg.norm(tracks[i] - detections[j])
            cost_matrix[i, j] = distance
    
    return cost_matrix


def visualize_assignment(cost_matrix, track_indices, detection_indices):
    """
    Visualize the assignment problem solution with actual 3D spatial positions
    """
    # Get the actual 3D positions of tracks and detections
    tracks, detections = get_3d_positions()
    
    plt.figure(figsize=(18, 8))
    
    # Plot cost matrix as heatmap
    plt.subplot(1, 3, 1)
    plt.imshow(cost_matrix, cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='3D Distance (m)')
    plt.title(f'Cost Matrix ({cost_matrix.shape[0]}x{cost_matrix.shape[1]})')
    plt.xlabel('Detection Index')
    plt.ylabel('Track Index')
    
    # Add text annotations only for smaller matrices
    if cost_matrix.shape[0] <= 8:
        for i in range(cost_matrix.shape[0]):
            for j in range(cost_matrix.shape[1]):
                plt.text(j, i, f'{cost_matrix[i, j]:.1f}', 
                        ha='center', va='center', 
                        color='black' if cost_matrix[i, j] < cost_matrix.max()/2 else 'white',
                        fontsize=8)
    
    # Highlight optimal assignment
    for track_idx, det_idx in zip(track_indices, detection_indices):
        plt.plot(det_idx, track_idx, 'bo', markersize=12, markerfacecolor='none', markeredgewidth=2)
    
    # Plot 3D spatial distribution (top-down view)
    plt.subplot(1, 3, 2)
    
    # Plot tracks (previous frame positions)
    track_colors = ['blue', 'green', 'purple']
    for i, (x, y, z) in enumerate(tracks):
        obj_type = i % 3
        plt.scatter(x, y, c=track_colors[obj_type], s=100, marker='s', alpha=0.7, 
                   label=f'Track {["Car", "Pedestrian", "Cyclist"][obj_type]}' if i < 3 else "")
        plt.text(x+0.5, y+0.5, f'T{i}', fontsize=8, ha='left')
    
    # Plot detections (current frame positions)  
    det_colors = ['red', 'orange', 'magenta']
    for i, (x, y, z) in enumerate(detections):
        obj_type = i % 3
        plt.scatter(x, y, c=det_colors[obj_type], s=100, marker='o', alpha=0.7,
                   label=f'Detection {["Car", "Pedestrian", "Cyclist"][obj_type]}' if i < 3 else "")
        plt.text(x+0.5, y-0.5, f'D{i}', fontsize=8, ha='left')
    
    # Draw assignment lines
    for track_idx, det_idx in zip(track_indices, detection_indices):
        track_pos = tracks[track_idx]
        det_pos = detections[det_idx]
        plt.plot([track_pos[0], det_pos[0]], [track_pos[1], det_pos[1]], 
                'lime', linewidth=2, alpha=0.8, linestyle='--')
        
        # Add cost label on assignment line
        mid_x = (track_pos[0] + det_pos[0]) / 2
        mid_y = (track_pos[1] + det_pos[1]) / 2
        distance = cost_matrix[track_idx, det_idx]
        plt.text(mid_x, mid_y, f'{distance:.1f}m', fontsize=7, ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.8))
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('3D Point Cloud Assignment\n(Top-down view)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Plot assignment graph (abstract view)
    plt.subplot(1, 3, 3)
    tracks_y = np.arange(len(track_indices))
    detections_y = np.arange(len(detection_indices))
    
    # Draw tracks on left
    plt.scatter(np.zeros(len(tracks_y)), tracks_y, c='blue', s=80, label='Tracks')
    for i, y in enumerate(tracks_y):
        plt.text(-0.1, y, f'T{i}', ha='right', va='center', fontsize=9)
    
    # Draw detections on right
    plt.scatter(np.ones(len(detections_y)), detections_y, c='red', s=80, label='Detections')
    for i, y in enumerate(detections_y):
        plt.text(1.1, y, f'D{i}', ha='left', va='center', fontsize=9)
    
    # Draw assignment lines
    for track_idx, det_idx in zip(track_indices, detection_indices):
        plt.plot([0, 1], [track_idx, det_idx], 'g-', linewidth=1.5, alpha=0.7)
        # Add cost label on line only for smaller matrices
        if len(track_indices) <= 8:
            mid_x, mid_y = 0.5, (track_idx + det_idx) / 2
            plt.text(mid_x, mid_y, f'{cost_matrix[track_idx, det_idx]:.1f}', 
                    ha='center', va='bottom', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
    
    plt.xlim(-0.3, 1.3)
    plt.ylim(-0.5, max(len(tracks_y), len(detections_y)) - 0.5)
    plt.title(f'Assignment Graph\n({len(track_indices)} assignments)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/hungarian_assignment_demo.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: output/hungarian_assignment_demo.png")
    plt.show()


def get_3d_positions():
    """
    Get the actual 3D positions of tracks and detections for visualization
    """
    # Create realistic 3D point cloud with random distribution
    n = 12
    tracks = []
    detections = []
    
    # Set same random seed for consistent results
    np.random.seed(42)
    
    # Generate tracks with realistic 3D point cloud distribution
    for i in range(n):
        # Random 3D positions within sensor range
        distance = np.random.uniform(5, 45)
        angle = np.random.uniform(0, 2 * np.pi)
        
        x = distance * np.cos(angle) + np.random.normal(0, 2.0)
        y = distance * np.sin(angle) + np.random.normal(0, 2.0)
        
        if i % 3 == 0:  # Cars
            z = np.random.uniform(0.5, 2.0)
        elif i % 3 == 1:  # Pedestrians  
            z = np.random.uniform(1.5, 1.8)
        else:  # Cyclists, trucks, etc.
            z = np.random.uniform(0.8, 2.5)
        
        tracks.append([x, y, z])
    
    # Generate detections with movement simulation
    for i, track_pos in enumerate(tracks):
        dt = 0.1  # 100ms between frames
        
        if i % 3 == 0:  # Cars
            velocity = np.random.uniform(0, 15)
        elif i % 3 == 1:  # Pedestrians
            velocity = np.random.uniform(0, 2)
        else:  # Cyclists, trucks
            velocity = np.random.uniform(0, 8)
        
        movement_angle = np.random.uniform(0, 2 * np.pi)
        
        dx = velocity * dt * np.cos(movement_angle)
        dy = velocity * dt * np.sin(movement_angle)
        dz = np.random.normal(0, 0.05)
        
        new_x = track_pos[0] + dx + np.random.normal(0, 0.3)
        new_y = track_pos[1] + dy + np.random.normal(0, 0.3)
        new_z = track_pos[2] + dz + np.random.normal(0, 0.1)
        
        detections.append([new_x, new_y, new_z])
    
    return np.array(tracks), np.array(detections)


def main():
    """
    Main function demonstrating Hungarian algorithm on realistic 3D point cloud data
    """
    # Create output directory
    import os
    os.makedirs('output', exist_ok=True)
    
    print("=== Hungarian Algorithm: Realistic 3D Point Cloud Tracking ===\n")
    print("Simulating KITTI-like autonomous vehicle object tracking scenario")
    print("with moving objects and realistic sensor measurements.\n")
    
    # Run the realistic 3D point cloud scenario
    extreme_results = create_extreme_scenario()
    
    # Create visualization of the realistic scenario
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATION")
    print("=" * 70)
    try:
        # Get the scenario data for visualization
        cost_matrix = get_extreme_scenario_data()
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)
        
        # Show assignment analysis
        print("\n🔍 ASSIGNMENT ANALYSIS:")
        print("Analyzing optimal vs nearest-neighbor assignments:")
        print("-" * 50)
        
        total_nearest_cost = 0
        assignment_examples = 0
        
        for i, (track_idx, det_idx) in enumerate(zip(track_indices, detection_indices)):
            if assignment_examples >= 4:  # Show first 4 examples
                break
            
            current_cost = cost_matrix[track_idx, det_idx]
            nearest_det = np.argmin(cost_matrix[track_idx, :])
            nearest_cost = cost_matrix[track_idx, nearest_det]
            total_nearest_cost += nearest_cost
            
            obj_type = ["Car", "Pedestrian", "Cyclist/Truck"][track_idx % 3]
            
            print(f"Track {track_idx} ({obj_type}):")
            print(f"  → Assigned to Detection {det_idx} (cost: {current_cost:.2f}m)")
            print(f"  → Nearest would be Detection {nearest_det} (cost: {nearest_cost:.2f}m)")
            
            if current_cost > nearest_cost + 0.1:  # Significant difference
                print(f"  ⚠️  Seems suboptimal (+{current_cost - nearest_cost:.2f}m)")
                if nearest_det in detection_indices:
                    conflicting_track = track_indices[np.where(detection_indices == nearest_det)[0][0]]
                    conflict_cost = cost_matrix[conflicting_track, nearest_det]
                    conflict_type = ["Car", "Pedestrian", "Cyclist/Truck"][conflicting_track % 3]
                    print(f"  🔍 But Detection {nearest_det} optimally assigned to Track {conflicting_track} ({conflict_type}, cost: {conflict_cost:.2f}m)")
                print(f"  ✅ Hungarian optimizes GLOBAL assignment across all objects")
            else:
                print(f"  ✅ This is the nearest/optimal detection")
            print()
            assignment_examples += 1
        
        visualize_assignment(cost_matrix, track_indices, detection_indices)
        print("✅ Realistic 3D point cloud visualization created successfully!")
        
    except ImportError:
        print("❌ matplotlib not available - skipping visualization")
    except Exception as e:
        print(f"❌ Visualization error: {e}")
    
    print("\n" + "=" * 70)
    print("HUNGARIAN ALGORITHM PRACTICE COMPLETE!")
    print("=" * 70)
    print("🎯 Key takeaways for KITTI object tracking:")
    print("1. Hungarian algorithm guarantees globally optimal assignment")
    print("2. Handles complex 3D point cloud scenarios with moving objects")
    print("3. Outperforms greedy nearest-neighbor in challenging scenarios")
    print("4. Essential for robust multi-object tracking in autonomous vehicles")
    print("5. Reduces ID switches and improves trajectory continuity")
    print("\n🚗 Ready for real-world KITTI dataset implementation!")


if __name__ == "__main__":
    main()
