# Estimate Pose between both images now
import cv2
import numpy as np


def estimate_pose(camera_matrix, mkpts0, mkpts1):
    E, mask = cv2.findEssentialMat(
        mkpts0,           # Keypoints from image 1
        mkpts1,           # Corresponding keypoints from image 2
        camera_matrix,
        method=cv2.RANSAC,
        prob=0.999,       # RANSAC confidence level
        threshold=1.0     # RANSAC inlier threshold (in pixels)
    )

    points, R, t, mask_pose = cv2.recoverPose(
            E,
            mkpts0,
            mkpts1,
            camera_matrix,
            mask=mask # Use the mask from findEssentialMat to only consider inliers
        )

    return R, t



def estimate_3d_points(mkpts0, mkpts1, camera_matrix, R, t):
    # Projection matrix for the first camera (at the origin)
    # It's K * [I | 0]
    P1 = camera_matrix @ np.hstack((np.identity(3), np.zeros((3, 1))))

    # Projection matrix for the second camera
    # It's K * [R | t]
    P2 = camera_matrix @ np.hstack((R, t))

    # --- Step 2: Triangulate 3D Points ---
    # OpenCV's triangulation function requires points to be in a specific format (2xN) and float type.
    points1_hom = mkpts0.T.astype(np.float32)
    points2_hom = mkpts1.T.astype(np.float32)

    # cv2.triangulatePoints returns points in homogeneous coordinates (4xN)
    points_4d_hom = cv2.triangulatePoints(P1, P2, points1_hom, points2_hom)

    # Convert from homogeneous to 3D coordinates by dividing by the 4th component
    points_3d = points_4d_hom / points_4d_hom[3]
    # We only need the first 3 rows (X, Y, Z)
    points_3d = points_3d[:3, :].T # Transpose to get N x 3

    return points_3d



def estimate_reprojection_error(mkpts0, mkpts1, camera_matrix, R, t):
    # Projection matrix for the first camera (at the origin)
    # It's K * [I | 0]
    P1 = camera_matrix @ np.hstack((np.identity(3), np.zeros((3, 1))))

    # Projection matrix for the second camera
    # It's K * [R | t]
    P2 = camera_matrix @ np.hstack((R, t))

    # --- Step 2: Triangulate 3D Points ---
    # OpenCV's triangulation function requires points to be in a specific format (2xN) and float type.
    points1_hom = mkpts0.T.astype(np.float32)
    points2_hom = mkpts1.T.astype(np.float32)

    # cv2.triangulatePoints returns points in homogeneous coordinates (4xN)
    points_4d_hom = cv2.triangulatePoints(P1, P2, points1_hom, points2_hom)

    # Convert from homogeneous to 3D coordinates by dividing by the 4th component
    points_3d = points_4d_hom / points_4d_hom[3]
    # We only need the first 3 rows (X, Y, Z)
    points_3d = points_3d[:3, :].T # Transpose to get N x 3

    # --- Step 3: Verify by Re-projecting the 3D points onto the second image ---

    # We need to use the full 4D homogeneous points for matrix multiplication
    # Re-project using the second camera's projection matrix P2
    projected_points2_hom = P2 @ points_4d_hom

    # Convert the projected points from homogeneous to 2D coordinates
    projected_points2 = projected_points2_hom / projected_points2_hom[2]
    # We only need the first 2 rows (x, y)
    projected_points2 = projected_points2[:2, :].T # Transpose to get N x 2

    # --- Calculate Reprojection Error ---
    # This is the average pixel distance between the original keypoints and the re-projected ones.
    # It's a great measure of how accurate your pose estimation and triangulation are.
    reprojection_error = np.mean(np.linalg.norm(mkpts1 - projected_points2, axis=1))

    # print(f"\nMean Reprojection Error: {reprojection_error:.4f} pixels")
    # A good result is typically less than 1.0 pixels.
    return reprojection_error