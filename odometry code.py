import numpy as np
import cv2
from tqdm  import tqdm
import math

def load_calib(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    
    # Extracting P0 matrix
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

# Apply lowe's ratio test
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

# Compute the Essential matrice from the fundamental matrix
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
    
    K, R_P, t_P,_,_,_,_ = cv2.decomposeProjectionMatrix(projection_matrix)
    t_P = t_P[:-1] /t_P[-1]
    
    R_f = R_P
    t_f = t_P
 
    traj = np.zeros((600, 600, 3), dtype=np.uint8)

    for i in tqdm(range(1, len(images))):
        keypoints1, descriptors1 = detect_features(images[i-1])
        keypoints2, descriptors2 = detect_features(images[i])

        good_matches = match_features(descriptors1,descriptors2)

        R,t = motion_estimation(good_matches,keypoints1,keypoints2,K)

        scale = scale_estimation(i,r'C:\Users\sharm\OneDrive\Desktop\SDC\poses\00.txt')
        
        t_f = t_f + 1*(R_f@t)
        R_f = R@R_f
    
        x = int(t_f[0][0] + 300)
        y = int(t_f[2][0] + 400)

        cv2.circle(traj, (x, y), 1, (0, 0, 255), 2)
        
        #cv2.putText(traj,f'x :{x} y : {y}',(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255))
        cv2.imshow('Trajectory', traj)
        cv2.imshow('Image Sequqence',images[i])
        cv2.waitKey(1)

def main():
    P0 = load_calib(r'C:\Users\sharm\OneDrive\Desktop\SDC\dataset\sequences\00\calib.txt')
    images = load_images(r'C:/Users/sharm/OneDrive/Desktop/SDC/dataset/sequences/00/image_0/',1000)
    poses = load_poses(r'C:\Users\sharm\OneDrive\Desktop\SDC\poses\00.txt',1000)

    visual_odometry(P0,images,poses)

main()

