#!/usr/bin/env python3
import os
import sys
import glob
import pickle
import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from scipy import optimize
import json

#############################################
#      Template-Based Scanning Classes      #
#############################################

with open("calibration.json", "r") as f:
    calib = json.load(f)

class Template:
    def __init__(self, img):
        if img is None:
            raise ValueError("Template initialization requires an image")
        self.img = img
        self.height = img.shape[0]
        self.width = img.shape[1]
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        # Real-world size in mm (adjust if needed)
        self.realH = 210  
        self.realW = 210  
        # Scale in m/pixel
        self.s = ((self.realH/self.height + self.realW/self.width)/2) / 1000.0
        self.circleList = []
        if len(self.img.shape) == 3:
            self.grayimg = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:
            self.grayimg = self.img
        self.pointcloud1 = o3d.geometry.PointCloud()
        self.pointcloud2 = o3d.geometry.PointCloud()

    def getPt(self):
        """Detect circles in the template image and create point clouds."""
        blurred = cv2.GaussianBlur(self.grayimg, (3, 3), 0)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=100,
            param2=30,
            minRadius=0,
            maxRadius=22
        )
        if circles is None:
            print("No circles detected in template.")
            return [], self.pointcloud1, self.pointcloud2
        circles = np.int32(np.around(circles))
        for i in circles[0, :]:
            self.circleList.append([i[0], i[1], i[2]])
        Pt1 = []
        for circle in self.circleList:
            # Compute 3D coordinates in template space (z=0)
            x = (circle[0] - self.cx) * self.s
            y = 0
            z = (circle[1] - self.cy) * self.s
            Pt1.append([x, y, z])
        self.pointcloud1.points = o3d.utility.Vector3dVector(np.array(Pt1))
        # Flip axes so that the template coordinate system becomes right-handed
        self.pointcloud1.transform([[1, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]])
        # Create a denser point cloud (neighbors)
        Pt2 = []
        for circle in self.circleList:
            for ix in range(-1, 2):
                for iz in range(-1, 2):
                    x = (circle[0] + ix - self.cx) * self.s
                    y = 0
                    z = (circle[1] + iz - self.cy) * self.s
                    Pt2.append([x, y, z])
        self.pointcloud2.points = o3d.utility.Vector3dVector(np.array(Pt2))
        self.pointcloud2.transform([[1, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]])
        return self.circleList, self.pointcloud1, self.pointcloud2

    def getFeature(self, n=3):
        """Extract geometric features (distances) from detected circles."""
        if not self.circleList:
            self.getPt()
        if len(self.circleList) < n:
            print(f"Warning: Only {len(self.circleList)} circles found, need at least {n}")
            return [], []
        feature_list = []
        new_xyr = []
        kd_tree = o3d.geometry.KDTreeFlann(self.pointcloud1)
        # Use all points from pointcloud1 for consistent ordering
        [_, idx1, _] = kd_tree.search_knn_vector_3d(self.pointcloud1.points[0], len(self.pointcloud1.points))
        points_array = np.asarray(self.pointcloud1.points)
        for point_i in idx1:
            try:
                [_, idx, _] = kd_tree.search_knn_vector_3d(points_array[point_i], n)
            except Exception as e:
                continue
            distances = np.array([])
            for i in range(n):
                for j in range(i+1, n):
                    a = points_array[idx[i]] - points_array[point_i]
                    b = points_array[idx[j]] - points_array[point_i]
                    d1 = np.linalg.norm(a)
                    d2 = np.linalg.norm(b)
                    d3 = np.linalg.norm(b - a)
                    distances = np.append(distances, [d1, d2, d3])
            feature_list.append([points_array[point_i], distances])
            new_xyr.append(self.circleList[point_i])
        return feature_list, new_xyr

def rigid_transform_3D(A, B):
    """
    Compute rigid transformation (rotation and translation)
    between two point sets A and B using SVD.
    """
    assert len(A) == len(B)
    A = A.T
    B = B.T
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)
    A_centered = A - centroid_A
    B_centered = B - centroid_B
    H = A_centered @ B_centered.T
    U, S, Vt = np.linalg.svd(H)
    det = np.linalg.det(Vt.T @ U.T)
    correction = np.eye(3)
    correction[2, 2] = det
    R = Vt.T @ correction @ U.T
    t = centroid_B - R @ centroid_A
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T

def fastMatch(sFeatureList, tFeatureList, distance_threshold=0.012):
    """
    Match features between source and target templates and
    compute the rigid transform from source to target.
    Returns the 4x4 transformation matrix.
    """
    match_list = []
    for sFeature in sFeatureList:
        best_dist = distance_threshold
        best_match = None
        for tFeature in tFeatureList:
            dist = np.sum(np.abs(sFeature[1] - tFeature[1]))
            if dist < best_dist:
                best_dist = dist
                best_match = tFeature[0]
        if best_match is not None:
            match_list.append((sFeature[0], best_match))
    if len(match_list) < 3:
        print("Not enough matches for a robust transformation.")
        return np.eye(4)
    A = np.array([m[0] for m in match_list])
    B = np.array([m[1] for m in match_list])
    T = rigid_transform_3D(A, B)
    return T

#############################################
#         Capture and Save Pipeline         #
#############################################

def capture_scans():
    # Create output folder if not exists
    nerf_folder = "./nerf_data"
    os.makedirs(nerf_folder, exist_ok=True)

    # Load reference template from a fixed location
    ref_template_path = "./template/template.png"
    if not os.path.exists(ref_template_path):
        print(f"Reference template not found at {ref_template_path}")
        return
    ref_img = cv2.imread(ref_template_path)
    if ref_img is None:
        print("Failed to load reference template image.")
        return
    ref_template = Template(ref_img)
    ref_template.getPt()
    ref_features, _ = ref_template.getFeature()

    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1920, 720, rs.format.z16, 30)
    profile = pipeline.start(config)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    fx = float(calib["rectified.0.fx"])
    fy = float(calib["rectified.0.fy"])
    cx = float(calib["rectified.0.ppx"])
    cy = float(calib["rectified.0.ppy"])
    width = int(calib["rectified.0.width"])
    height = int(calib["rectified.0.height"])
    # Save intrinsics as a 3x3 matrix
    intrinsic_matrix = np.array([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0, 0, 1]])
    np.save(os.path.join(nerf_folder, "intrinsics.npy"), intrinsic_matrix)
    print("Camera intrinsics saved.")

    poses = []
    img_counter = 0

    print("Press 'c' to capture a view, 'q' to quit.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            # Optionally, display the current frame
            cv2.imshow("Live Capture", color_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                # Capture current view
                print(f"Capturing view {img_counter} ...")
                # Create a Template instance for the captured image
                curr_template = Template(color_image)
                circles, _ = curr_template.getPt()
                if len(circles) < 3:
                    print("Not enough circles detected in current frame; skipping capture.")
                    continue
                curr_features, _ = curr_template.getFeature()
                # Compute rigid transformation from reference to current frame
                T = fastMatch(ref_features, curr_features)
                # Invert T so that we store a camera-to-world matrix (assuming reference = world)
                cam_pose = np.linalg.inv(T)
                poses.append(cam_pose)
                # Resize image to 400x400 for consistency
                saved_img = cv2.resize(color_image, (400, 400))
                img_filename = os.path.join(nerf_folder, f"color_{img_counter:04d}.png")
                cv2.imwrite(img_filename, saved_img)
                print(f"Saved {img_filename}")
                img_counter += 1
            elif key == ord('q'):
                print("Exiting capture loop.")
                break
    except Exception as e:
        print(f"Error during capture: {e}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    if len(poses) == 0:
        print("No valid poses captured.")
        return
    poses = np.stack(poses, axis=0)  # shape (N, 4, 4)
    np.save(os.path.join(nerf_folder, "poses.npy"), poses)
    print(f"Saved {len(poses)} poses to {os.path.join(nerf_folder, 'poses.npy')}")

#############################################
#                Main Entry                 #
#############################################

def main():
    """
    Run the capture pipeline.
    Press 'c' to capture a view.
    Press 'q' to quit and save all captured data.
    The script outputs the captured color images (400x400 PNGs),
    the poses.npy file, and intrinsics.npy into the ./nerf_data folder.
    """
    capture_scans()

if __name__ == "__main__":
    main()