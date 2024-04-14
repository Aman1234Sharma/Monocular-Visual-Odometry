import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

# Load KITTI dataset
def load_images(path, num_images): # Function takes image_path and number of images to load.
    images = []  # list in which images are stored iteratively
    for i in range(num_images):   #loop runs num_images times 
        img_path = f"{path}/{str(i).zfill(6)}.png" # path to each image in image_o folder is created
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) #image is read in grayscale format
        images.append(img) #  image is append to the list
    return images  # after all images are loaded, containing list is returned

# Load calibration parameters
def load_calibration(calib_path): #Function to load calibration file from path
    with open(calib_path, 'r') as f: #Opens the calibration file located at calib_path in read mode ('r'), accesed by name f.
        lines = f.readlines() #Reads all lines from the file and stores them in the lines list.
    
    # Extracting P0 matrix values from the first line after removing the 'P0: ' prefix
    P0_values = lines[0].strip().split(' ')[1:]
    
    # Converting the list of values to a NumPy array of type float32
    P0 = np.array(P0_values, dtype=np.float32).reshape(3, 4)  # Reshaping to 3x4 matrix
    
    return P0

# Load ground truth poses
def load_poses(poses_path, num_poses):  
    poses = [] 
    with open(poses_path, 'r') as f: 
        lines = f.readlines()
    
    for i in range(num_poses):
        pose = np.array([float(x) for x in lines[i].strip().split(' ')[0:]])
        poses.append(pose)
    
    return np.array(poses)

# Feature detection and description using ORB
def detect_and_describe_features(image):
    orb = cv2.ORB_create(5000) # an ORB object is created with 3000 keypoints
    keypoints, descriptors = orb.detectAndCompute(image, None) #keypoints and descriptors calculated 
    return keypoints, descriptors


# Match features between two consecutive images using RANSAC
def match_features(descriptors1, descriptors2, keypoints1, keypoints2, img1, img2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    draw_matches(img1, keypoints1, img2, keypoints2, good_matches)
    
    # Compute essential matrix using RANSAC
    E, mask = cv2.findEssentialMat(src_pts, dst_pts, focal=718.856, pp=(607.1928,185.2157), method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    # Recover relative pose
    _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts)  
    
    good_matches = [m for m, msk in zip(good_matches, mask) if msk[0] == 1]
    
    return good_matches, R, t

def draw_matches(img1, keypoints1, img2, keypoints2, matches):
 # Resize images to have the same height
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Calculate the new dimensions
    new_height = max(h1, h2)
    new_width1 = int(w1 * new_height / h1)
    new_width2 = int(w2 * new_height / h2)
    
    # Resize the images
    img1_resized = cv2.resize(img1, (new_width1, new_height))
    img2_resized = cv2.resize(img2, (new_width2, new_height))
    
    # Concatenate the images side by side
    img_matches = np.hstack((img1_resized, img2_resized))
    
    # Draw matches
    img_matches = cv2.drawMatches(img1_resized, keypoints1, img2_resized, keypoints2, matches, None)
    
    cv2.imshow('Matches', img_matches)
    #cv2.waitKey(1) # Wait for any key press
    cv2.destroyAllWindows()

def extend_matrix(R, t):
    """
    Extend the rotation matrix and translation vector to 4x4 transformation matrix.
    
    Parameters:
    - R: Rotation matrix
    - t: Translation vector
    
    Returns:
    - T: Extended 4x4 transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T

def triangulate_points(keypoints1, keypoints2, P0, T1, T2):
    """
    Triangulate 3D points from matched keypoints and camera projection matrices.
    
    Parameters:
    - keypoints1: keypoints from the first frame
    - keypoints2: keypoints from the second frame
    - P0: camera projection matrix of the first frame
    - T1: 4x4 transformation matrix of the first frame
    - T2: 4x4 transformation matrix of the second frame
    
    Returns:
    - points_3d: 3D triangulated points
    """
    R1 = T1[:3, :3]
    t1 = T1[:3, 3]
    R2 = T2[:3, :3]
    t2 = T2[:3, 3]

    P1 = np.dot(P0[:, :3], np.hstack((R1, t1.reshape(3, 1))))
    P2 = np.dot(P0[:, :3], np.hstack((R2, t2.reshape(3, 1))))
    points_4d = cv2.triangulatePoints(P1[:3], P2[:3], keypoints1.T, keypoints2.T)
    points_3d = points_4d[:3] / points_4d[3]
    return points_3d.T
# Visual odometry
def visual_odometry(images, calib_path):
    trajectory = [np.zeros(3),]
    R = np.eye(3)
    t = np.zeros((3, 1))
    
    P0 = load_calibration(calib_path)
    
    for i in tqdm(range(1, len(images))):
        keypoints1, descriptors1 = detect_and_describe_features(images[i-1])
        keypoints2, descriptors2 = detect_and_describe_features(images[i])
        
        matches, R_delta, t_delta = match_features(descriptors1, descriptors2, keypoints1, keypoints2, images[i-1], images[i])
        
        R = np.dot(R, R_delta)
        t = t + np.dot(R, t_delta)

        # Extend transformation matrices to 4x4
        T1 = extend_matrix(R_delta, t_delta)
        T2 = extend_matrix(R, t)
        
        # Triangulate 3D points
        points_3d = triangulate_points(np.float32([keypoint.pt for keypoint in keypoints1]), 
                                       np.float32([keypoint.pt for keypoint in keypoints2]), 
                                       P0, T1, T2)


        trajectory.append(t.flatten())
    
    return np.array(trajectory)

def plot_trajectory(trajectory, gt_poses):
    plt.figure()
    plt.plot(trajectory[:, 0], trajectory[:, 2], label='Estimated')
    plt.plot(gt_poses[:, 3], -gt_poses[:, 10], label='Ground Truth')
    plt.title('Visual Odometry Trajectory')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.legend()
    plt.draw()
    plt.pause(100000)

if __name__ == "__main__":

    # Directories defined
    dataset_path = r'C:/Users/sharm/OneDrive/Desktop/SDC/dataset/sequences/00/image_0/'
    calib_path = r'C:\Users\sharm\OneDrive\Desktop\SDC\dataset\sequences\00\calib.txt'
    poses_path = r'C:\Users\sharm\OneDrive\Desktop\SDC\poses\00.txt'
    
    num_images = 100  # number of images to process
    
    # Load sequence
    images = load_images(dataset_path, num_images)
    
    # Run visual odometry
    trajectory = visual_odometry(images, calib_path)
    
    # Load ground truth poses
    gt_poses = load_poses(poses_path, num_images)
    
    plot_trajectory(np.array(trajectory), gt_poses)
    
    

