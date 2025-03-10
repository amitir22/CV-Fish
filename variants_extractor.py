"""
optical_flow_module.py

Example module that computes various optical flow results using OpenCV.
Requires:
    pip install opencv-python opencv-contrib-python
"""

import cv2
import numpy as np
from typing import Dict, Any

def compute_optical_flow_variants(frame1: np.ndarray, frame2: np.ndarray) -> Dict[str, Any]:
    """
    Computes multiple optical flow variants (dense and sparse) from two consecutive frames.
    
    Args:
        frame1 (np.ndarray): First video frame (e.g., BGR color from OpenCV).
        frame2 (np.ndarray): Second video frame (e.g., BGR color from OpenCV).
        
    Returns:
        Dict[str, Any]: Dictionary containing results from different optical flow algorithms.
    """
    # Convert frames to grayscale (most flow algorithms expect grayscale input)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 1. Dense Optical Flow (Farneback)
    # --------------------------------------------------
    # Default parameters (tune as needed)
    farneback_flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    # 2. Dense Optical Flow (TV-L1) – requires opencv-contrib
    # --------------------------------------------------
    # If you have opencv-contrib, you can use:
    #   cv2.optflow.calcOpticalFlowDual_TVL1(gray1, gray2, None)
    # Otherwise, comment this block out or handle the import carefully
    try:
        tvl1_flow = cv2.optflow.calcOpticalFlowDual_TVL1(gray1, gray2, None)
    except AttributeError:
        tvl1_flow = None  # Fallback if `cv2.optflow` isn't present

    # 3. Sparse Optical Flow (Lucas–Kanade, calcOpticalFlowPyrLK)
    # --------------------------------------------------
    # 3a. Detect good features in the first frame
    max_corners = 100
    corners = cv2.goodFeaturesToTrack(gray1, maxCorners=max_corners, qualityLevel=0.3, minDistance=7)
    
    # 3b. Compute optical flow for those feature points
    if corners is not None:
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            gray1, 
            gray2, 
            corners, 
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
    else:
        next_pts, status, err = None, None, None

    # 4. Construct results dictionary
    # --------------------------------------------------
    results = {
        "dense_farneback": farneback_flow,  # 2D flow vectors for every pixel
        "dense_tvl1": tvl1_flow,           # 2D flow vectors for every pixel (if available)
        "sparse_pyr_lk": {
            "prev_pts": corners,           # Points in first frame
            "next_pts": next_pts,          # Points in second frame
            "status": status,              # Status (1 if flow found, 0 otherwise)
            "error": err                   # Error measure for each point
        }
    }

    return results


# Example usage (uncomment to run a quick test):
# if __name__ == "__main__":
#     # Load two sample frames here for testing:
#     # frame1 = cv2.imread("frame1.png")
#     # frame2 = cv2.imread("frame2.png")
#     # flows = compute_optical_flow_variants(frame1, frame2)
#     # print(flows.keys())
#     pass
