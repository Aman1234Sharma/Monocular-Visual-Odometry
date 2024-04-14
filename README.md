# Monocular-Visual-Odometry
Monocular visual odometry is a technique used to estimate the motion of a camera in a 3D environment using only a single camera. It relies on feature tracking and matching between consecutive frames to compute the camera's pose and trajectory over time.

## Monocular Visual Odometry Pipeline
### 1. Image Acquisition:

Capture images from a single camera at regular intervals or from a video stream.We get these image sequence directly from kitti dataset. 

### Pre-processing:

Convert images to grayscale or apply color correction.
Undistort images using camera calibration parameters if available.
Kitti dataset sequence is already in grayscale,undistorted and rectified, and callibration matrix is provided.
Feature Detection and Tracking:

Detect keypoints (e.g., corners, edges) in the current image.
Track these keypoints across consecutive frames using feature matching techniques (e.g., optical flow).
Motion Estimation:

Compute the relative pose change between consecutive frames using the matched keypoints.
Estimate the essential matrix or fundamental matrix using the matched keypoints.
Depth Estimation (Optional):

Triangulate keypoints to estimate their 3D positions.
Compute depth information using stereo vision or monocular depth estimation techniques.
Scale Estimation (Optional):

Estimate the scale factor to recover the true motion scale.
Use additional sensors (e.g., IMU, wheel encoders) or scene constraints to improve scale estimation.
Loop Closure and Optimization (Optional):

Detect loop closures to correct drift and improve the map consistency.
Perform bundle adjustment or pose graph optimization to refine the trajectory and map.
Visualization and Mapping:

Visualize the estimated camera trajectory and 3D map in real-time or post-processing.
Store the trajectory and map data for further analysis or integration with other systems.
Evaluation and Validation:

Evaluate the accuracy and robustness of the visual odometry system using ground truth data or benchmark datasets.
Validate the performance under various conditions (e.g., indoor, outdoor, different lighting).
