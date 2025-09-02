"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley

    Fixed and formatted for modern Python (2025)
"""

from __future__ import print_function
import os
import glob
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # safer backend, works headless
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from filterpy.kalman import KalmanFilter

np.random.seed(0)


# ---------------- Linear Assignment ---------------- #
def linear_assignment(cost_matrix):
    """Assign detections to trackers using Hungarian (or LAPJV if installed)."""
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array(list(zip(x, y)))
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


# ---------------- IoU Calculation ---------------- #
def iou_batch(bb_test, bb_gt):
    """
    Computes IoU between two sets of boxes [x1,y1,x2,y2].
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / (
        (bb_test[..., 2] - bb_test[..., 0]) *
        (bb_test[..., 3] - bb_test[..., 1]) +
        (bb_gt[..., 2] - bb_gt[..., 0]) *
        (bb_gt[..., 3] - bb_gt[..., 1]) - wh
    )
    return o


# ---------------- BBox Conversions ---------------- #
def convert_bbox_to_z(bbox):
    """[x1,y1,x2,y2] -> [x,y,s,r] (center, scale, aspect)."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """[x,y,s,r] -> [x1,y1,x2,y2]."""
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2.,
                         x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2.,
                         x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


# ---------------- Kalman Tracker ---------------- #
class KalmanBoxTracker:
    """Represents an individual tracked object with a Kalman filter."""

    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4, 0] = convert_bbox_to_z(bbox).flatten()
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """Update state with new bbox."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """Advance state and return predicted bbox."""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """Return current bbox estimate."""
        return convert_x_to_bbox(self.kf.x)


# ---------------- Data Association ---------------- #
def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """Assign detections to trackers based on IoU."""
    if len(trackers) == 0:
        return (np.empty((0, 2), dtype=int),
                np.arange(len(detections)),
                np.empty((0,), dtype=int))

    iou_matrix = iou_batch(detections, trackers)
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty((0, 2))

    unmatched_detections = [d for d in range(len(detections))
                            if d not in matched_indices[:, 0]]
    unmatched_trackers = [t for t in range(len(trackers))
                          if t not in matched_indices[:, 1]]

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# ---------------- SORT Tracker ---------------- #
class Sort:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
            dets - [[x1,y1,x2,y2,score], ...]
        Returns:
            [[x1,y1,x2,y2,id], ...]
        """
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold)

        # update matched trackers
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and \
               (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))


# ---------------- CLI ---------------- #
def parse_args():
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', action='store_true',
                        help='Display tracker output (slow)')
    parser.add_argument("--seq_path", type=str, default='data',
                        help="Path to detections.")
    parser.add_argument("--phase", type=str, default='train',
                        help="Subdirectory in seq_path.")
    parser.add_argument("--max_age", type=int, default=1,
                        help="Max #frames to keep a track without detections.")
    parser.add_argument("--min_hits", type=int, default=3,
                        help="Min detections before track is valid.")
    parser.add_argument("--iou_threshold", type=float, default=0.3,
                        help="Minimum IoU for match.")
    return parser.parse_args()


# ---------------- Main ---------------- #
if __name__ == '__main__':
    args = parse_args()
    display = args.display
    phase = args.phase
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3)

    if display:
        if not os.path.exists('mot_benchmark'):
            print("\nERROR: 'mot_benchmark' link not found!\n")
            exit()
        plt.ion()
        fig, ax1 = plt.subplots()

    if not os.path.exists('output'):
        os.makedirs('output')

    pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
    for seq_dets_fn in glob.glob(pattern):
        mot_tracker = Sort(max_age=args.max_age,
                           min_hits=args.min_hits,
                           iou_threshold=args.iou_threshold)
        seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
        seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]

        with open(os.path.join('output', f'{seq}.txt'), 'w') as out_file:
            print(f"Processing {seq}.")
            for frame in range(int(seq_dets[:, 0].max())):
                frame += 1
                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                dets[:, 2:4] += dets[:, 0:2]  # [x,y,w,h] -> [x1,y1,x2,y2]
                total_frames += 1

                if display:
                    fn = os.path.join('mot_benchmark', phase, seq,
                                      'img1', f'{frame:06d}.jpg')
                    im = io.imread(fn)
                    ax1.clear()
                    ax1.imshow(im)
                    plt.title(seq + ' Tracked Targets')

                start_time = time.time()
                trackers = mot_tracker.update(dets)
                cycle_time = time.time() - start_time
                total_time += cycle_time

                for d in trackers:
                    print(f"{frame},{d[4]},{d[0]:.2f},{d[1]:.2f},"
                          f"{(d[2]-d[0]):.2f},{(d[3]-d[1]):.2f},1,-1,-1,-1",
                          file=out_file)
                    if display:
                        d = d.astype(np.int32)
                        ax1.add_patch(patches.Rectangle(
                            (d[0], d[1]), d[2]-d[0], d[3]-d[1],
                            fill=False, lw=3, ec=colours[d[4] % 32, :]
                        ))

                if display:
                    plt.pause(0.001)

    print("Total Tracking took: %.3f seconds for %d frames (%.1f FPS)" %
          (total_time, total_frames, total_frames / total_time))

    if display:
        print("Note: run without --display for real runtime results.")
