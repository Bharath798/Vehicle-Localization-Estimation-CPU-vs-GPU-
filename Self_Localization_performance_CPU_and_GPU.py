#!/usr/bin/env python3
"""
vehicle_shift_cpu_gpu.py

Updated: estimates the 2D position (cumulative image-plane translation) of a tracked
vehicle across frames using feature matching + RANSAC-based affine estimation.

Features:
  - CPU pipeline (ORB + BFMatcher + cv2.estimateAffinePartial2D with RANSAC)
  - GPU pipeline (uses cv2.cuda for feature detection/matching if available). The
    affine estimation currently runs on CPU (we upload/download descriptors/keypoints
    as needed) because OpenCV's RANSAC estimators live on CPU in most builds.
  - Per-frame translation (dx, dy), number of inliers, and cumulative position
    (cum_x, cum_y) saved to CSVs for CPU and GPU runs.
  - Optional overlay video output showing arrows and cumulative position on frames.
  - Timing instrumentation for CPU vs GPU comparison and a saved performance plot.

Usage:
    python vehicle_shift_cpu_gpu.py --input input.mov --max-frames 300 --resize-width 800

Dependencies:
    pip install numpy opencv-python matplotlib

GPU notes:
  * GPU requires an OpenCV build with CUDA support (cv2.cuda). If not available,
    GPU run will be skipped gracefully.

Author: Generated for user
"""

import argparse
import sys
import time
from pathlib import Path
import csv

import numpy as np
import cv2
import matplotlib.pyplot as plt
import cv2
print(cv2.__version__)


def extract_frames(video_path, max_frames=None, resize_width=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if resize_width is not None:
            h, w = frame.shape[:2]
            if w != resize_width:
                new_h = int(h * (resize_width / w))
                frame = cv2.resize(frame, (resize_width, new_h))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        yield idx, frame, gray
        idx += 1
        if max_frames is not None and idx >= max_frames:
            break
    cap.release()


# -------------------- CPU implementation ---------------------------------

def detect_and_compute_cpu(orb, frame_gray):
    kp = orb.detect(frame_gray, None)
    kp, des = orb.compute(frame_gray, kp)
    return kp, des


def match_cpu(matcher, des1, des2, ratio_test=True, knn_k=2):
    if des1 is None or des2 is None:
        return []
    if ratio_test:
        matches_knn = matcher.knnMatch(des1, des2, k=knn_k)
        good = []
        for m in matches_knn:
            if len(m) == 2:
                a, b = m
                if a.distance < 0.75 * b.distance:
                    good.append(a)
        return sorted(good, key=lambda x: x.distance)
    else:
        matches = matcher.match(des1, des2)
        return sorted(matches, key=lambda x: x.distance)


# -------------------- GPU implementation (cv2.cuda) ----------------------

def is_cuda_available():
    return hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0


def detect_and_compute_gpu(orb_gpu, frame_gray):
    gmat = cv2.cuda_GpuMat()
    gmat.upload(frame_gray)
    # kp_gpu = orb_gpu.detectAndComputeAsync(gmat)
    kp_gpu, descriptors_gpu = orb_gpu.detectAndComputeAsync(gmat, mask=None)
    # convert keypoints list from GPU object (some OpenCV versions differ)
    try:
        kp = orb_gpu.convert(kp_gpu)
    except Exception:
        # some builds provide getKeypoints or return CPU-friendly structure
        kp = [
            cv2.KeyPoint(x=float(k.pt[0]), y=float(k.pt[1]), _size=k.size,
                         _angle=k.angle, _response=k.response, _octave=getattr(k, 'octave', 0),
                         _class_id=getattr(k, 'class_id', -1))
            for k in kp_gpu
        ]
    # compute descriptors
    # _, des_gpu = orb_gpu.compute(gmat, kp_gpu)
    # des = des_gpu.download() if des_gpu is not None else None
    des = descriptors_gpu.download() if descriptors_gpu is not None else None
    # Convert kp to list of cv2.KeyPoint if necessary
    kp_cpu = []
    for k in kp:
        try:
            kp_cpu.append(cv2.KeyPoint(x=float(k.pt[0]), y=float(k.pt[1]), _size=k.size,
                                      _angle=k.angle, _response=k.response,
                                      _octave=getattr(k, 'octave', 0), _class_id=getattr(k, 'class_id', -1)))
        except Exception:
            # fallback: if kp is already python KeyPoint-like
            kp_cpu.append(k)
    return kp_cpu, des


def match_gpu(matcher_gpu, des1, des2):
    if des1 is None or des2 is None:
        return []
    gdes1 = cv2.cuda_GpuMat()
    gdes2 = cv2.cuda_GpuMat()
    gdes1.upload(des1)
    gdes2.upload(des2)
    matches = matcher_gpu.match(gdes1, gdes2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


# -------------------- Estimation & Utility -------------------------------

def matches_to_points(matches, kp1, kp2):
    pts1 = []
    pts2 = []
    for m in matches:
        if m.queryIdx < len(kp1) and m.trainIdx < len(kp2):
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
    if len(pts1) == 0:
        return None, None
    pts1 = np.array(pts1, dtype=np.float32)
    pts2 = np.array(pts2, dtype=np.float32)
    return pts1, pts2


def estimate_affine_translation(pts1, pts2, ransac_thresh=3.0):
    # returns dx, dy, num_inliers, affine_matrix
    if pts1 is None or pts2 is None or len(pts1) < 3:
        return 0.0, 0.0, 0, None
    M, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
    if M is None:
        return 0.0, 0.0, 0, None
    dx = float(M[0, 2])
    dy = float(M[1, 2])
    nin = int(np.sum(inliers)) if inliers is not None else 0
    return dx, dy, nin, M


# -------------------- Runner functions ----------------------------------

def run_cpu_pipeline(frames, save_overlay=False, overlay_out=None):
    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    times = []
    records = []  # tuples: (frame_from, frame_to, dx, dy, ninliers, cum_x, cum_y)
    cum_x = 0.0
    cum_y = 0.0

    prev_kp = prev_des = prev_frame = None
    prev_idx = None

    writer = None

    for idx, frame_color, frame_gray in frames:
        t0 = time.perf_counter()
        kp, des = detect_and_compute_cpu(orb, frame_gray)
        if prev_des is not None:
            # match
            matches = bf.knnMatch(prev_des, des, k=2)
            good = []
            for m in matches:
                if len(m) == 2:
                    a, b = m
                    if a.distance < 0.75 * b.distance:
                        good.append(a)
            pts1, pts2 = matches_to_points(good, prev_kp, kp)
            dx, dy, nin, M = estimate_affine_translation(pts1, pts2)
            cum_x += dx
            cum_y += dy
            records.append((prev_idx, idx, dx, dy, nin, cum_x, cum_y))
            # overlay
            if save_overlay:
                if writer is None and overlay_out is not None:
                    h, w = frame_color.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(overlay_out, fourcc, 20.0, (w, h))
                vis = frame_color.copy()
                # draw arrow from center shifted by dx,dy
                cx, cy = w // 2, h // 2
                end = (int(cx + cum_x), int(cy + cum_y))
                cv2.arrowedLine(vis, (cx, cy), end, (0, 255, 0), 2, tipLength=0.2)
                cv2.putText(vis, f"cum_x={cum_x:.1f}, cum_y={cum_y:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                writer.write(vis)
        else:
            records.append((None, idx, 0.0, 0.0, 0, cum_x, cum_y))
        t1 = time.perf_counter()
        times.append(t1 - t0)

        prev_kp, prev_des, prev_frame, prev_idx = kp, des, frame_gray, idx

    if writer is not None:
        writer.release()
    return records, times


def run_gpu_pipeline(frames, save_overlay=False, overlay_out=None):
    if not is_cuda_available():
        raise RuntimeError('cv2.cuda not available or no CUDA device')

    # create GPU ORB and BF matcher
    try:
        orb_gpu = cv2.cuda.ORB_create(nfeatures=2000)
    except Exception:
        # fallback names
        orb_gpu = cv2.cuda_ORB.create(nfeatures=2000)

    matcher_gpu = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_HAMMING)

    times = []
    records = []
    cum_x = 0.0
    cum_y = 0.0

    prev_kp = prev_des = prev_frame = None
    prev_idx = None
    writer = None

    for idx, frame_color, frame_gray in frames:
        t0 = time.perf_counter()
        kp, des = detect_and_compute_gpu(orb_gpu, frame_gray)
        if prev_des is not None:
            # GPU matcher requires uploading descriptors
            matches = match_gpu(matcher_gpu, prev_des, des)
            pts1, pts2 = matches_to_points(matches, prev_kp, kp)
            dx, dy, nin, M = estimate_affine_translation(pts1, pts2)
            cum_x += dx
            cum_y += dy
            records.append((prev_idx, idx, dx, dy, nin, cum_x, cum_y))
            if save_overlay:
                if writer is None and overlay_out is not None:
                    h, w = frame_color.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(overlay_out, fourcc, 20.0, (w, h))
                vis = frame_color.copy()
                cx, cy = w // 2, h // 2
                end = (int(cx + cum_x), int(cy + cum_y))
                cv2.arrowedLine(vis, (cx, cy), end, (0, 0, 255), 2, tipLength=0.2)
                cv2.putText(vis, f"cum_x={cum_x:.1f}, cum_y={cum_y:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                writer.write(vis)
        else:
            records.append((None, idx, 0.0, 0.0, 0, cum_x, cum_y))
        t1 = time.perf_counter()
        times.append(t1 - t0)
        prev_kp, prev_des, prev_frame, prev_idx = kp, des, frame_gray, idx
    if writer is not None:
        writer.release()
    return records, times


# -------------------- Plotting & saving --------------------------------

def plot_performance(cpu_times, gpu_times, out_path):
    plt.figure(figsize=(10,5))
    x_cpu = np.arange(len(cpu_times))
    x_gpu = np.arange(len(gpu_times))
    plt.plot(x_cpu, np.cumsum(cpu_times), label=f'CPU (total {sum(cpu_times):.2f}s)')
    if len(gpu_times) > 0:
        plt.plot(x_gpu, np.cumsum(gpu_times), label=f'GPU (total {sum(gpu_times):.2f}s)')
    plt.xlabel('frame index')
    plt.ylabel('cumulative processing time (s)')
    plt.title('CPU vs GPU cumulative processing time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_records_csv(records, fname):
    with open(fname, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame_from', 'frame_to', 'dx', 'dy', 'n_inliers', 'cum_x', 'cum_y'])
        for r in records:
            w.writerow(r)

def plot_positions(cpu_positions, gpu_positions, out_path):
    """
    Plot vehicle positions (X, Y) for CPU and GPU pipelines.

    Args:
        cpu_positions (list of tuple): [(x1, y1), (x2, y2), ...] for CPU
        gpu_positions (list of tuple): [(x1, y1), (x2, y2), ...] for GPU
        out_path (str): path to save the plot
    """
    plt.figure(figsize=(8, 8))

    # Unpack X, Y
    if len(cpu_positions) > 0:
        _,_,_,_,_,cpu_x, cpu_y = zip(*cpu_positions)
        plt.plot(cpu_x, cpu_y, 'r-', marker='o', label='CPU trajectory')

    if len(gpu_positions) > 0:
        _,_,_,_,_,gpu_x, gpu_y = zip(*gpu_positions)
        plt.plot(gpu_x, gpu_y, 'b-', marker='x', label='GPU trajectory')

    plt.xlabel('X position (pixels)')
    plt.ylabel('Y position (pixels)')
    plt.title('CPU vs GPU Estimated Vehicle Position')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # keep aspect ratio 1:1 for trajectory
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -------------------- CLI & orchestration -------------------------------

def main():
    p = argparse.ArgumentParser(description='Estimate vehicle position shifts (CPU vs GPU)')
    p.add_argument('--input', '-i', required=True, help='.mov input video')
    p.add_argument('--max-frames', type=int, default=300)
    p.add_argument('--resize-width', type=int, default=800)
    p.add_argument('--skip-gpu', action='store_true')
    p.add_argument('--out-plot', default='cpu_gpu_performance.png')
    p.add_argument('--save-overlay', action='store_true', help='save overlay video(s) showing cumulative shift')
    p.add_argument('--overlay-cpu', default='cpu_overlay.mp4')
    p.add_argument('--overlay-gpu', default='gpu_overlay.mp4')
    args = p.parse_args()

    video_path = Path(args.input)
    if not video_path.exists():
        print('Input not found:', video_path)
        sys.exit(1)

    print('Loading frames into memory (this uses RAM) ...')
    frames = list(extract_frames(video_path, max_frames=args.max_frames, resize_width=args.resize_width))
    # frames: list of (idx, color_frame, gray_frame)
    print(f'Loaded {len(frames)} frames')

    # Prepare iterators for CPU (we pass color+gray tuples)
    frames_for_cpu = [(i, color, gray) for (i, color, gray) in frames]

    print('Running CPU pipeline...')
    t0 = time.perf_counter()
    cpu_records, cpu_times = run_cpu_pipeline(frames_for_cpu, save_overlay=args.save_overlay, overlay_out=args.overlay_cpu if args.save_overlay else None)
    t1 = time.perf_counter()
    print(f'CPU total processing time: {sum(cpu_times):.3f}s (script wall {t1-t0:.3f}s)')

    gpu_records = []
    gpu_times = []
    # if not args.skip_gpu:
    try:
        if not is_cuda_available():
            print('cv2.cuda not available or no GPU detected; skipping GPU run')
        else:
            print('Running GPU pipeline...')
            frames_for_gpu = [(i, color, gray) for (i, color, gray) in frames]
            t0g = time.perf_counter()
            gpu_records, gpu_times = run_gpu_pipeline(frames_for_gpu, save_overlay=args.save_overlay, overlay_out=args.overlay_gpu if args.save_overlay else None)
            t1g = time.perf_counter()
            print(f'GPU total processing time: {sum(gpu_times):.3f}s (script wall {t1g-t0g:.3f}s)')
    except Exception as e:
        print('GPU pipeline failed:', e)
        print('Proceeding without GPU results.')

    # Save CSVs
    save_records_csv(cpu_records, 'cpu_shifts.csv')
    if gpu_records:
        save_records_csv(gpu_records, 'gpu_shifts.csv')

    # Plot performance
    plot_performance(cpu_times, gpu_times, args.out_plot)
    print('Saved performance plot to', args.out_plot)
    print('Saved cpu_shifts.csv', 'and gpu_shifts.csv' if gpu_records else '')
    plot_positions(cpu_records, gpu_records, "positions.png")

    print('Sample CPU records (first 10):')
    for r in cpu_records[:10]:
        print(r)
    if gpu_records:
        print('Sample GPU records (first 10):')
        for r in gpu_records[:10]:
            print(r)


if __name__ == '__main__':
    main()
