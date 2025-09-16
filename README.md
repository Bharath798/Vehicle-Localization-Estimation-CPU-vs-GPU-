# Vehicle Shift Estimation (CPU vs GPU) 🚗
This project estimates the 2D position (cumulative image-plane translation) of a tracked vehicle across video frames. It uses feature matching with the ORB algorithm and RANSAC-based affine estimation to determine the vehicle's movement. It includes both a CPU and a GPU pipeline for performance comparison, and it provides visualizations of the trajectory and the final output.

⚙️ Features
Feature Matching: Utilizes ORB (Oriented FAST and Rotated BRIEF) and RANSAC for robust motion estimation.

CPU Pipeline: Implements the core logic using ORB, BFMatcher, and cv2.estimateAffinePartial2D.

GPU Pipeline: Leverages cv2.cuda for GPU-accelerated ORB detection and matching, requiring a CUDA-enabled OpenCV build.

Per-Frame Shift Estimation:

Extracts the translation (dx, dy) for each pair of consecutive frames.

Calculates the cumulative position (cum_x, cum_y).

Records the number of inlier matches for each frame.

Outputs:

CSV logs of per-frame translations for both CPU and GPU.

Overlay videos showing the cumulative vehicle trajectory drawn on the video frames.

Plots comparing CPU vs. GPU runtime performance and estimated trajectories.

🛠️ Dependencies
Python 3.7+

Required packages:
```
pip install numpy opencv-python matplotlib
```

GPU Requirements:

A CUDA-enabled device.

An OpenCV build with CUDA support (check if cv2.cuda is available).

🚀 Usage
Run the script from the command line:

Bash
```
python vehicle_shift_cpu_gpu.py --input input.mov --max-frames 300 --resize-width 800
```

Arguments:
--input, -i: Path to the input video file (.mov, .mp4, etc.). [required]

--max-frames: Maximum number of frames to process (default: 300).

--resize-width: Resize width for processing frames (default: 800).

--skip-gpu: Skips the GPU pipeline, even if CUDA is available.

--out-plot: Path to save the CPU vs. GPU performance plot (default: cpu_gpu_performance.png).

--save-overlay: Saves overlay videos visualizing the trajectory arrows.

--overlay-cpu: Output filename for the CPU trajectory overlay video (default: cpu_overlay.mp4).

--overlay-gpu: Output filename for the GPU trajectory overlay video (default: gpu_overlay.mp4).

Example:
Bash
```
python vehicle_shift_cpu_gpu.py --input highway.mov --max-frames 200 --save-overlay
```

📁 Output Files
After running the script, the following files will be generated:

Trajectory Data
cpu_shifts.csv: Contains the CPU results with per-frame (dx, dy, n_inliers, cum_x, cum_y).

gpu_shifts.csv: Contains the GPU results (if the GPU pipeline was run).

Plots
cpu_gpu_performance.png: A plot comparing the CPU and GPU runtimes.

positions.png: A plot showing the CPU vs. GPU estimated trajectory paths in the XY-plane.

📈 Results and Analysis
After processing a test video of 300 frames, both the CPU and GPU implementations generated per-frame translation estimates, cumulative trajectory plots, and runtime comparisons. The findings are summarized below.

Runtime Performance
CPU pipeline: ~3.75 seconds total processing time.

GPU pipeline: ~4.10 seconds total processing time.

Although the GPU version processes feature detection and matching in parallel, the total runtime was slightly slower than the CPU version. This is because additional time was spent on data transfer between CPU and GPU memory, particularly for the affine estimation, which is not GPU-accelerated in this implementation. This suggests that the GPU's performance advantage may become more apparent with larger videos, higher-resolution frames, or a greater number of features to track.

Accuracy of Translation Estimates
Both pipelines estimated similar per-frame translations (dx, dy) and cumulative positions (cum_x, cum_y).

CPU Trajectory: The motion was gradual and consistent, with 800–1200 inlier matches per frame, indicating stable tracking.

GPU Trajectory: The GPU results showed slightly larger per-frame translations and a higher average number of inlier matches (1200–1450 range). This suggests that the GPU-accelerated ORB detected more features, leading to slightly different, but still consistent, motion estimates.
