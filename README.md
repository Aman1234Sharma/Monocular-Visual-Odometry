# Monocular-Visual-Odometry
Monocular visual odometry is a technique used to estimate the motion of a camera in a 3D environment using only a single camera. It relies on feature tracking and matching between consecutive frames to compute the camera's pose and trajectory over time.

## Monocular Visual Odometry Pipeline
### 1. Image Acquisition:
* Capture images from a single camera at regular intervals or from a video stream.
* I got these image sequence directly from kitti dataset. 

### 2. Pre-processing:
* Convert images to grayscale or apply color correction.
* Undistort images using camera calibration parameters.
* Kitti dataset sequence is already in grayscale,undistorted and rectified, and callibration matrix is also provided.

### 3. Feature Detection and Tracking:
* Detect keypoints in the current image. I use ORB Feature detector for this purpose.
* Track these keypoints across consecutive frames. I used FLANN matcher for this.
  
### 4. Motion Estimation:
* Compute the relative pose change between consecutive frames using the matched keypoints.
* Estimate the essential matrix or fundamental matrix using the matched keypoints.

### 5. Scale Estimation:
* Estimate the scale factor to recover the true motion scale.
* I have used ground truth poses for estimating absolute scale.

### 6. Visualization,Evaluation and Validation:
* Visualize the estimated camera trajectory and 3D map in real-time.
* Store the trajectory and map data for further analysis or integration with other systems.
* Evaluate the accuracy and robustness of the visual odometry system using ground truth data.
* Validate the performance under various conditions (e.g., indoor, outdoor, different lighting).

## Techstacks
* Programming Language - Python
* Libraries used - numpy,cv2,matplotlib,tqdm,mpl_toolkit,math
## References
Avi Singh's Blog - [Monocular Visual Odometry using OpenCV](https://avisingh599.github.io/vision/monocular-vo/)
Some Basics - (https://www.maths.lth.se/matematiklth/personal/calle/datorseende13/notes/forelas1.pdf)
