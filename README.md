"# Vehicle Shift Estimation (CPU vs GPU)-"
This project estimates the 2D position (cumulative image-plane translation) of a tracked vehicle across video frames using feature matching and RANSAC-based affine estimation. It provides a CPU and GPU pipeline for performance comparison, trajectory visualization, and overlay outputs.

Features
Feature Matching with ORB and RANSAC

CPU pipeline: ORB + BFMatcher + cv2.estimateAffinePartial2D

GPU pipeline: ORB detection/matching with cv2.cuda (if CUDA-enabled OpenCV exists)

Per-frame shift estimation

Extracts translation (dx, dy) for every frame pair

Computes cumulative position (cum_x, cum_y)

Records number of inlier matches

Output

CSV logs of per-frame translations for CPU and GPU

Overlay videos showing cumulative vehicle trajectory drawn on frames

Performance comparison plot for CPU vs GPU runtime

Trajectory plot comparing CPU vs GPU estimated motion


Dependencies
Python 3.7+

Required packages:

bash
pip install numpy opencv-python matplotlib
GPU Requirements:

A CUDA-enabled device
OpenCV build with CUDA (cv2.cuda must be available)


Usage
Run the script from the command line:
python vehicle_shift_cpu_gpu.py --input input.mov --max-frames 300 --resize-width 800

Arguments:
--input, -i : Path to input video file (.mov, .mp4, etc.) [required]

--max-frames : Maximum number of frames to process (default: 300)

--resize-width : Resize width for processing frames (default: 800)

--skip-gpu : Skip GPU pipeline even if CUDA is available

--out-plot : Path to save CPU vs GPU performance plot (default: cpu_gpu_performance.png)

--save-overlay : Save overlay videos visualizing trajectory arrows

--overlay-cpu : Output filename for CPU trajectory overlay video (default: cpu_overlay.mp4)

--overlay-gpu : Output filename for GPU trajectory overlay video (default: gpu_overlay.mp4)


Output Files
After running, the following files are generated:

Trajectory data

cpu_shifts.csv — CPU results with per-frame (dx, dy, n_inliers, cum_x, cum_y)

gpu_shifts.csv — GPU results (if GPU available)

Plots

cpu_gpu_performance.png — Runtime comparison plot

positions.png — CPU vs GPU trajectory paths in XY-plane

Example
python vehicle_shift_cpu_gpu.py --input highway.mov --max-frames 200 --save-overlay
