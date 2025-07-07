#!/usr/bin/env python3
"""
Multi-Object Tracker with consistent top-down visualization, legend, and GIF output

Features:
1. Kalman-based multi-object tracking with nearest-neighbor association
2. Consistent frame scaling: fixed axis limits
3. Top-down view of LiDAR clusters with bounding-box overlays
4. Animated GIF generation
5. Legend indicating cluster points, tentative and confirmed tracks

Dependencies:
- numpy
- kalman_filter module
- multi_frame_pipeline module
- matplotlib
- imageio
- os
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import imageio
import os
from matplotlib.lines import Line2D

from kalman_filter import KalmanFilter
from multi_frame_pipeline import MultiFramePerceptionPipeline, Detection, FrameData


class TrackState(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    DELETED = "deleted"


@dataclass
class Track:
    id: int
    kalman_filter: KalmanFilter
    state: TrackState
    hits: int
    time_since_update: int
    age: int
    last_detection: Optional[Detection] = None

    def __post_init__(self):
        self.creation_time = time.time()

    def get_position(self) -> np.ndarray:
        return self.kalman_filter.get_position()

    def get_velocity(self) -> np.ndarray:
        return self.kalman_filter.get_velocity()

    def predict(self):
        self.kalman_filter.predict()
        self.time_since_update += 1
        self.age += 1

    def update(self, detection: Detection):
        self.kalman_filter.update(detection.center)
        self.hits += 1
        self.time_since_update = 0
        self.last_detection = detection
        if self.state == TrackState.TENTATIVE and self.hits >= 3:
            self.state = TrackState.CONFIRMED

    def mark_missed(self):
        self.time_since_update += 1
        self.age += 1
        if self.time_since_update > 5:
            self.state = TrackState.DELETED


class MultiObjectTracker:
    def __init__(self, max_distance=5.0, dt=0.1):
        self.max_distance = max_distance
        self.dt = dt
        self.tracks: List[Track] = []
        self.track_id_counter = 0
        self.frame_count = 0
        self.track_history: Dict[int, List[np.ndarray]] = {}
        self.processing_times: List[float] = []

    def predict(self):
        for track in self.tracks:
            track.predict()

    def _cost_matrix(self, dets: List[Detection]) -> np.ndarray:
        if not self.tracks or not dets: return np.array([])
        M, N = len(self.tracks), len(dets)
        cost = np.zeros((M, N))
        for i, t in enumerate(self.tracks):
            p = t.get_position()
            for j, d in enumerate(dets): cost[i, j] = np.linalg.norm(p - d.center)
        return cost

    def _associate(self, dets: List[Detection]):
        if not self.tracks: return [], list(range(len(dets))), []
        if not dets: return [], [], list(range(len(self.tracks)))
        cost = self._cost_matrix(dets)
        matches, u_d, u_t = [], list(range(len(dets))), list(range(len(self.tracks)))
        for ti in list(u_t):
            best = min(u_d, key=lambda di: cost[ti, di], default=None)
            if best is not None and cost[ti, best] < self.max_distance:
                matches.append((ti, best))
                u_d.remove(best)
                u_t.remove(ti)
        return matches, u_d, u_t

    def _init_tracks(self, dets, unmatched):
        for di in unmatched:
            d = dets[di]
            kf = KalmanFilter(dt=self.dt)
            kf.initialize(d.center)
            tr = Track(self.track_id_counter, kf, TrackState.TENTATIVE, 1, 0, 1, d)
            self.tracks.append(tr)
            self.track_history[self.track_id_counter] = [d.center.copy()]
            self.track_id_counter += 1

    def _update_tracks(self, dets, matches):
        for ti, di in matches:
            self.tracks[ti].update(dets[di])
            self.track_history[self.tracks[ti].id].append(dets[di].center.copy())

    def _cleanup(self, unmatched_t):
        for ti in unmatched_t: self.tracks[ti].mark_missed()
        self.tracks = [t for t in self.tracks if t.state != TrackState.DELETED]

    def update(self, dets: List[Detection]):
        start = time.time()
        self.predict()
        matches, u_d, u_t = self._associate(dets)
        self._update_tracks(dets, matches)
        self._cleanup(u_t)
        self._init_tracks(dets, u_d)
        self.frame_count += 1
        self.processing_times.append(time.time() - start)

    def stats(self):
        active = sum(1 for t in self.tracks if t.state==TrackState.CONFIRMED)
        print(f"Frame {self.frame_count}: total={len(self.tracks)}, confirmed={active}")


class TrackingVisualizer:
    def __init__(self, tracker: MultiObjectTracker, xlim: Tuple[float,float], ylim: Tuple[float,float]):
        self.tracker = tracker
        self.colors = plt.cm.Set3(np.linspace(0,1,20))
        self.xlim, self.ylim = xlim, ylim

    def plot(self, ax, dets: List[Detection], idx: int):
        ax.clear()
        # Plot clusters
        for d in dets:
            pts = d.points; ax.scatter(pts[:,0], pts[:,1], s=2, alpha=0.2, color='gray')
        # Plot boxes & tracks
        for t in self.tracker.tracks:
            d = t.last_detection
            if not d: continue
            cx, cy, _ = d.center; lx, ly, _ = d.dimensions
            x0, y0 = cx-lx/2, cy-ly/2
            clr = self.colors[t.id % len(self.colors)]
            rect = plt.Rectangle((x0,y0), lx, ly,
                                 ec=clr, fc='none', lw=2 if t.state==TrackState.CONFIRMED else 1,
                                 ls='-' if t.state==TrackState.CONFIRMED else '--')
            ax.add_patch(rect)
            px, py = t.get_position()[:2]
            marker = 'o' if t.state==TrackState.CONFIRMED else '^'
            ax.scatter(px, py, c=[clr], s=80, marker=marker, edgecolors='black')
            ax.text(px, py+0.2, f'T{t.id}', ha='center', va='bottom', color=clr)
            v = t.get_velocity()[:2]
            if np.linalg.norm(v)>0.1: ax.arrow(px,py,v[0],v[1],head_width=0.1,head_length=0.1,fc=clr,ec=clr)
        # Legend
        legend_elements = [
            Line2D([0],[0], marker='.', color='gray', label='Cluster points', markersize=5, linestyle='None', alpha=0.5),
            Line2D([0],[0], marker='^', color='black', label='Tentative track', markersize=10, linestyle='None'),
            Line2D([0],[0], marker='o', color='black', label='Confirmed track', markersize=10, linestyle='None')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        # Fixed view
        ax.set_xlim(*self.xlim); ax.set_ylim(*self.ylim)
        ax.set_aspect('equal', 'box')
        ax.set_title(f"Frame {idx}")
        ax.set_xlabel('X'); ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)


def main():
    data_path = "data/2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync"
    cfg = {'filter':{'min_volume':0.5,'min_points':15}, 'dbscan':{'eps':0.8,'min_points':8}}
    pipeline = MultiFramePerceptionPipeline(data_path, cfg)
    tracker = MultiObjectTracker(max_distance=3.0, dt=0.1)

    # Fixed axis limits
    xlim = (-90, 90); ylim = (-40, 60)
    viz = TrackingVisualizer(tracker, xlim, ylim)
    out_dir = "output/tracking"; os.makedirs(out_dir, exist_ok=True)
    frames = []
    start, end = 1, 100
    for i in range(start, end+1):
        fr = pipeline.process_single_frame(i)
        tracker.update(fr.detections)
        tracker.stats()
        fig, ax = plt.subplots(figsize=(12,8))
        viz.plot(ax, fr.detections, i)
        plt.tight_layout()
        path = os.path.join(out_dir, f"frm_{i}.png")
        fig.savefig(path, dpi=150); plt.close(fig)
        frames.append(imageio.imread(path))
    gif_path = os.path.join(out_dir, 'tracking.gif')
    imageio.mimsave("tracking.gif", frames, duration=0.5, loop=0)
    print(f"Saved GIF: {gif_path}")

if __name__=='__main__': main()
