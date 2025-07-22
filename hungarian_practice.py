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


def create_sample_cost_matrix():
    """
    Create a sample cost matrix representing distances between
    3 tracks and 3 detections
    
    Returns:
        cost_matrix: 3x3 matrix where cost[i][j] is the distance
                    between track i and detection j
    """
    # Sample positions for 3 existing tracks (x, y, z)
    tracks = np.array([
        [10.0, 5.0, 0.5],   # Track 0
        [15.0, 8.0, 0.3],   # Track 1
        [20.0, 12.0, 0.7]   # Track 2
    ])
    
    # Sample positions for 3 new detections (x, y, z)
    detections = np.array([
        [10.5, 5.2, 0.4],   # Detection 0 - close to Track 0
        [19.8, 12.3, 0.6],  # Detection 1 - close to Track 2
        [14.7, 8.1, 0.2]    # Detection 2 - close to Track 1
    ])
    
    # Calculate Euclidean distances
    cost_matrix = np.zeros((len(tracks), len(detections)))
    
    for i, track_pos in enumerate(tracks):
        for j, det_pos in enumerate(detections):
            # Euclidean distance in 3D
            distance = np.linalg.norm(track_pos - det_pos)
            cost_matrix[i, j] = distance
    
    return cost_matrix, tracks, detections


def demonstrate_greedy_vs_hungarian():
    """
    Compare greedy nearest-neighbor assignment vs Hungarian algorithm
    """
    print("=" * 70)
    print("HUNGARIAN ALGORITHM PRACTICE - ASSIGNMENT PROBLEM")
    print("=" * 70)
    
    # Create sample data
    cost_matrix, tracks, detections = create_sample_cost_matrix()
    
    print("\n1. SAMPLE DATA:")
    print("-" * 30)
    print("Track positions:")
    for i, pos in enumerate(tracks):
        print(f"  Track {i}: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]")
    
    print("\nDetection positions:")
    for i, pos in enumerate(detections):
        print(f"  Detection {i}: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]")
    
    print("\n2. COST MATRIX (Euclidean distances):")
    print("-" * 40)
    print("        Det0    Det1    Det2")
    for i in range(len(cost_matrix)):
        row_str = f"Track{i}  "
        for j in range(len(cost_matrix[i])):
            row_str += f"{cost_matrix[i, j]:.3f}   "
        print(row_str)
    
    # Greedy nearest-neighbor approach (current implementation)
    print("\n3. GREEDY NEAREST-NEIGHBOR ASSIGNMENT:")
    print("-" * 45)
    greedy_assignments = []
    used_detections = set()
    total_greedy_cost = 0
    
    for track_idx in range(len(tracks)):
        min_cost = float('inf')
        best_detection = -1
        
        for det_idx in range(len(detections)):
            if det_idx not in used_detections:
                cost = cost_matrix[track_idx, det_idx]
                if cost < min_cost:
                    min_cost = cost
                    best_detection = det_idx
        
        if best_detection != -1:
            greedy_assignments.append((track_idx, best_detection))
            used_detections.add(best_detection)
            total_greedy_cost += min_cost
            print(f"  Track {track_idx} -> Detection {best_detection} (cost: {min_cost:.3f})")
    
    print(f"  Total greedy cost: {total_greedy_cost:.3f}")
    
    # Hungarian algorithm (optimal assignment)
    print("\n4. HUNGARIAN ALGORITHM (OPTIMAL) ASSIGNMENT:")
    print("-" * 50)
    
    # Use scipy's implementation of the Hungarian algorithm
    track_indices, detection_indices = linear_sum_assignment(cost_matrix)
    
    total_hungarian_cost = 0
    for track_idx, det_idx in zip(track_indices, detection_indices):
        cost = cost_matrix[track_idx, det_idx]
        total_hungarian_cost += cost
        print(f"  Track {track_idx} -> Detection {det_idx} (cost: {cost:.3f})")
    
    print(f"  Total Hungarian cost: {total_hungarian_cost:.3f}")
    
    # Compare results
    print("\n5. COMPARISON:")
    print("-" * 20)
    print(f"Greedy total cost:    {total_greedy_cost:.3f}")
    print(f"Hungarian total cost: {total_hungarian_cost:.3f}")
    improvement = total_greedy_cost - total_hungarian_cost
    print(f"Improvement:          {improvement:.3f} ({improvement/total_greedy_cost*100:.1f}%)")
    
    if improvement > 0.001:  # Small threshold for floating point comparison
        print("✅ Hungarian algorithm found a better assignment!")
    else:
        print("ℹ️  Both algorithms found the same optimal solution.")
    
    return cost_matrix, track_indices, detection_indices


def demonstrate_challenging_case():
    """
    Demonstrate a case where greedy fails but Hungarian succeeds
    """
    print("\n" + "=" * 70)
    print("CHALLENGING CASE - WHERE GREEDY FAILS")
    print("=" * 70)
    
    # Create a scenario where greedy makes a suboptimal choice
    tracks = np.array([
        [0.0, 0.0, 0.0],   # Track 0 at origin
        [5.0, 0.0, 0.0],   # Track 1 at (5,0,0)
        [10.0, 0.0, 0.0]   # Track 2 at (10,0,0)
    ])
    
    detections = np.array([
        [1.0, 0.0, 0.0],   # Detection 0 - close to Track 0
        [4.0, 0.0, 0.0],   # Detection 1 - between Track 0 and 1
        [10.1, 0.0, 0.0]   # Detection 2 - very close to Track 2
    ])
    
    # Calculate cost matrix
    cost_matrix = np.zeros((len(tracks), len(detections)))
    for i, track_pos in enumerate(tracks):
        for j, det_pos in enumerate(detections):
            distance = np.linalg.norm(track_pos - det_pos)
            cost_matrix[i, j] = distance
    
    print("\nCost matrix:")
    print("        Det0    Det1    Det2")
    for i in range(len(cost_matrix)):
        row_str = f"Track{i}  "
        for j in range(len(cost_matrix[i])):
            row_str += f"{cost_matrix[i, j]:.3f}   "
        print(row_str)
    
    # Greedy assignment
    greedy_cost = 0
    used = set()
    print("\nGreedy assignment:")
    for track_idx in range(len(tracks)):
        min_cost = float('inf')
        best_det = -1
        for det_idx in range(len(detections)):
            if det_idx not in used and cost_matrix[track_idx, det_idx] < min_cost:
                min_cost = cost_matrix[track_idx, det_idx]
                best_det = det_idx
        if best_det != -1:
            used.add(best_det)
            greedy_cost += min_cost
            print(f"  Track {track_idx} -> Detection {best_det} (cost: {min_cost:.3f})")
    
    # Hungarian assignment
    track_indices, detection_indices = linear_sum_assignment(cost_matrix)
    hungarian_cost = cost_matrix[track_indices, detection_indices].sum()
    
    print("\nHungarian assignment:")
    for track_idx, det_idx in zip(track_indices, detection_indices):
        cost = cost_matrix[track_idx, det_idx]
        print(f"  Track {track_idx} -> Detection {det_idx} (cost: {cost:.3f})")
    
    print(f"\nGreedy total:    {greedy_cost:.3f}")
    print(f"Hungarian total: {hungarian_cost:.3f}")
    print(f"Improvement:     {greedy_cost - hungarian_cost:.3f}")


def visualize_assignment(cost_matrix, track_indices, detection_indices):
    """
    Visualize the assignment problem solution
    """
    plt.figure(figsize=(10, 6))
    
    # Plot cost matrix as heatmap
    plt.subplot(1, 2, 1)
    plt.imshow(cost_matrix, cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='Cost (distance)')
    plt.title('Cost Matrix')
    plt.xlabel('Detection Index')
    plt.ylabel('Track Index')
    
    # Add text annotations
    for i in range(cost_matrix.shape[0]):
        for j in range(cost_matrix.shape[1]):
            plt.text(j, i, f'{cost_matrix[i, j]:.2f}', 
                    ha='center', va='center', color='black' if cost_matrix[i, j] < cost_matrix.max()/2 else 'white')
    
    # Highlight optimal assignment
    for track_idx, det_idx in zip(track_indices, detection_indices):
        plt.plot(det_idx, track_idx, 'bo', markersize=15, markerfacecolor='none', markeredgewidth=3)
    
    # Plot assignment graph
    plt.subplot(1, 2, 2)
    tracks_y = np.arange(len(track_indices))
    detections_y = np.arange(len(detection_indices))
    
    # Draw tracks on left
    plt.scatter(np.zeros(len(tracks_y)), tracks_y, c='blue', s=100, label='Tracks')
    for i, y in enumerate(tracks_y):
        plt.text(-0.1, y, f'T{i}', ha='right', va='center')
    
    # Draw detections on right
    plt.scatter(np.ones(len(detections_y)), detections_y, c='red', s=100, label='Detections')
    for i, y in enumerate(detections_y):
        plt.text(1.1, y, f'D{i}', ha='left', va='center')
    
    # Draw assignment lines
    for track_idx, det_idx in zip(track_indices, detection_indices):
        plt.plot([0, 1], [track_idx, det_idx], 'g-', linewidth=2, alpha=0.7)
        # Add cost label on line
        mid_x, mid_y = 0.5, (track_idx + det_idx) / 2
        plt.text(mid_x, mid_y, f'{cost_matrix[track_idx, det_idx]:.2f}', 
                ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
    
    plt.xlim(-0.3, 1.3)
    plt.ylim(-0.5, max(len(tracks_y), len(detections_y)) - 0.5)
    plt.title('Optimal Assignment')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/hungarian_assignment_demo.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: output/hungarian_assignment_demo.png")
    plt.show()


def main():
    """
    Main function demonstrating Hungarian algorithm usage
    """
    # Create output directory
    import os
    os.makedirs('output', exist_ok=True)
    
    # Demonstrate basic usage
    cost_matrix, track_indices, detection_indices = demonstrate_greedy_vs_hungarian()
    
    # Show challenging case
    demonstrate_challenging_case()
    
    # Create visualization
    try:
        visualize_assignment(cost_matrix, track_indices, detection_indices)
    except ImportError:
        print("\nNote: Install matplotlib to see visualizations")
    
    print("\n" + "=" * 70)
    print("PRACTICE COMPLETE!")
    print("=" * 70)
    print("Key takeaways:")
    print("1. Hungarian algorithm guarantees optimal assignment")
    print("2. Greedy approach can be suboptimal in complex scenarios")
    print("3. scipy.optimize.linear_sum_assignment implements Hungarian algorithm")
    print("4. Simply pass cost matrix and get optimal indices")


if __name__ == "__main__":
    main()
