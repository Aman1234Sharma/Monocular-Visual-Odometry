import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm  import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def load_calib(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    
    # Extracting P0 matrix values from the first line after removing the 'P0: ' prefix
    P0_values = lines[0].strip().split(' ')[1:]
    
    # Converting the list of values to a NumPy array of type float32
    P0 = np.array(P0_values, dtype=np.float32).reshape(3, 4)  # Reshaping to 3x4 matrix
    
    return P0
def load_images(image_path,num_images):
    images = []
    for i in range(num_images):
        img_path = f"{image_path}/{str(i).zfill(6)}.png"
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        images.append(img)
    return images

def load_poses(poses_path, num_poses):
    poses = []
    with open(poses_path, 'r') as f:
        lines = f.readlines()
    
    for i in range(num_poses):
        pose = np.array([float(x) for x in lines[i].strip().split(' ')[0:]])
        poses.append(pose)
    
    return np.array(poses)

def detect_features(image):
    orb = cv2.ORB_create(3000)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(descriptors1,descriptors2):

# FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)

# Initialize FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

# Match descriptors
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return good_matches

def motion_estimation(good_matches,keypoints1,keypoints2,K):
# Extract matching keypoints
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Compute the fundamental matrix
    fundamental_matrix, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_LMEDS)

# Compute the camera matrices from the fundamental matrix
    E = np.matmul(np.matmul(K.T, fundamental_matrix), K)

# Decompose the essential matrix to obtain the rotation and translation
    _, R, t, _ = cv2.recoverPose(E, points1, points2, K)

    return(R,t)

def scale_estimation(frame_id,path):
    i = 0
    x, y, z = 0, 0, 0
    x_prev, y_prev, z_prev = 0, 0, 0
    
    with open(path, "r") as myfile:
        for line in myfile:
            if i > frame_id:
                break
            
            z_prev = z
            x_prev = x
            y_prev = y
            
            values = line.strip().split()
            for j, value in enumerate(values):
                if j == 7:
                    y = float(value)
                if j == 3:
                    x = float(value)
                if j == 11:
                    z = float(value)
            
            i += 1
    
    return math.sqrt((x - x_prev)**2 + (y - y_prev)**2 + (z - z_prev)**2)

def visual_odometry(projection_matrix,images,ground_truth_poses):
    
    K, R, t,_,_,_,_ = cv2.decomposeProjectionMatrix(projection_matrix)
    t = t[:-1] /t[-1]
    
    R_cam = R
    t_cam = t

    R_cam_list = [np.eye(3)]  
    t_cam_list = [np.zeros((3, 1))]
 

    for i in tqdm(range(1, len(images))):
        keypoints1, descriptors1 = detect_features(images[i-1])
        keypoints2, descriptors2 = detect_features(images[i])

        good_matches = match_features(descriptors1,descriptors2)

        R,t = motion_estimation(good_matches,keypoints1,keypoints2,K)
        scale = scale_estimation(i,r'C:\Users\sharm\OneDrive\Desktop\SDC\poses\00.txt')

        #if scale > 0.1 and t[2] > t[0] and t[2] > t[1]:
        R_cam = R@R_cam
        t_cam = t + scale*(t.T@R_cam).T

        R_cam_list.append(R_cam)
        t_cam_list.append(t_cam)

    return R_cam_list,t_cam_list

def triangulate_points(points1,points2):
    
    points1_reshaped = points1.reshape(-1, 2)
    points2_reshaped = points2.reshape(-1, 2)
    
    
    # Perform triangulation
    points_3d_homogeneous = cv2.triangulatePoints(P0, P, points1_reshaped.T, points2_reshaped.T)
    
    # Convert homogeneous coordinates to 3D coordinates
    points_3d = points_3d_homogeneous[:3] / points_3d_homogeneous[3]
    
    # Reshape to get 3D coordinates
    points_3d = points_3d.reshape(-1, 3)
    
    print("3D Coordinates of Tracked Features:")
    print(points_3d)

    return points_3d

def main():
    P0 = load_calib(r'C:\Users\sharm\OneDrive\Desktop\SDC\dataset\sequences\00\calib.txt')
    images = load_images(r'C:/Users/sharm/OneDrive/Desktop/SDC/dataset/sequences/00/image_0/',20)
    poses = load_poses(r'C:\Users\sharm\OneDrive\Desktop\SDC\poses\00.txt',20)

    R,t = visual_odometry(P0,images,poses)
    R_cam_array = np.array(R)
    #print(R_cam_array)
    t_cam_array = np.array([np.array(t) for t in t])

