#!/usr/bin/env python3
"""
variants_extractor.py
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple
from frozendict import frozendict

# NOTE: if you want to edit the params, edit their copy with:
#       >>> editable_params = dict(DEFAULT_<*>_PARAMS)
DEFAULT_FARNEBACK_PARAMS = frozendict({
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 15,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0
})

DEFAULT_TVL1_PARAMS = frozendict({})

DEFAULT_LUCAS_KANADE_PARAMS = frozendict({
    "win_size": (15, 15),
    "max_level": 2,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    "max_corners": 100,
    "quality_level": 0.3,
    "min_distance": 7,
    "block_size": 7
})


def to_gray(frame1: np.ndarray, frame2: np.ndarray):
    """
    Converts the 2 frames to gray-scale.

    Parameters:
        frame1 (numpy.ndarray): The first (non-gray scale) frame.
        frame2 (numpy.ndarray): The second (non-gray scale) frame.

    Returns:
        tuple: (gray1, gray2)
    """
    gray1 =  cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 =  cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    return gray1, gray2


def extract_farneback_variant(frame1: np.ndarray, frame2: np.ndarray, 
                                pyr_scale: float, levels: int, winsize: int, 
                                iterations: int, poly_n: int, poly_sigma: float,
                                flags: int):
    """
    Extracts the farneback variants from the calculated flow between frame1 and frame2.

    Parameters:
        frame1 (numpy.ndarray): The first (non-gray scale) frame.
        frame2 (numpy.ndarray): The second (non-gray scale) frame.

        pyr_scale (float): The pyr_scale param for farneback.
        levels (int): The levels param for farneback.
        winsize (int): The winsize param for farneback.
        iterations (int): The iterations param for farneback.
        poly_n (int): The poly_n param for farneback.
        poly_sigma (float): The poly_sigma param for farneback.
        flags (int): The flags param for farneback.

    Returns:
        dict: A dictionary containing the magnitude sum, magnitude deviation, and angular deviation
              of the farneback flow calculation.
    """
    gray1, gray2 = to_gray(frame1, frame2)

    farneback_flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=pyr_scale,
        levels=levels,
        winsize=winsize,
        iterations=iterations,
        poly_n=poly_n,
        poly_sigma=poly_sigma,
        flags=flags
    )

    # aggregate result values: magnitude_sum, angular_deviation, magnitude_deviation
    return calculate_flow_metrics(farneback_flow)


def extract_TVL1_variant(frame1: np.ndarray, frame2: np.ndarray):
    """
    Extracts the TVL1 variants from the calculated flow between frame1 and frame2.

    Parameters:
        frame1 (numpy.ndarray): The first (non-gray scale) frame.
        frame2 (numpy.ndarray): The second (non-gray scale) frame.

    Returns:
        dict: A dictionary containing the magnitude sum, magnitude deviation, and angular deviation
              of the TVL1 flow calculation.
    """
    gray1, gray2 = to_gray(frame1, frame2)

    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = tvl1.calc(gray1, gray2, None)

    # aggregate result values: magnitude_sum, angular_deviation, magnitude_deviation
    return calculate_flow_metrics(flow)


def extract_lucas_kanade_variant(frame1: np.ndarray, frame2: np.ndarray, 
                                    win_size: Tuple[int, int], max_level: int,
                                    criteria: Tuple[int, int, float], max_corners: int,
                                    quality_level: float, min_distance: int, 
                                    block_size: int):
    """
    Extracts the lucas kanade variants from the calculated flow between frame1 and frame2.

    Parameters:
        frame1 (numpy.ndarray): The first (non-gray scale) frame.
        frame2 (numpy.ndarray): The second (non-gray scale) frame.

        win_size (Tuple[int, int]): The winSize param for lucas kanade.
        max_level (int): The maxLevel param for lucas kanade.
        criteria (Tuple[int, int, float]): The opencv criteria params for lucas kanade.

        max_corners (int): The maxCorners param for good feature tracking.
        quality_level (float): The qualityLevel param for good feature tracking. 
        min_distance (int): The minDistance param for good feature tracking.
        block_size (int): The blockSize param for good feature tracking. 

    Returns:
        dict: A dictionary containing the magnitude sum, magnitude deviation, and angular deviation
              of the Lucas-Kanade flow calculation.
    """
    gray1, gray2 = to_gray(frame1, frame2)

    # init lucas kanade params
    lk_params = dict()
    lk_params['winSize'] = win_size
    lk_params['maxLevel'] = max_level
    lk_params['criteria'] = criteria

    # init good features to track params
    feature_params = dict()
    feature_params['maxCorners'] = max_corners
    feature_params['qualityLevel'] = quality_level
    feature_params['minDistance'] = min_distance
    feature_params['blockSize'] = block_size

    # detect and keep good features
    p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)

    # calc optical flow between 2 frames
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

    # filter valid points:
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # complete the dense flow array from sparse values
    h, w = gray1.shape

    # 2 in (h, w, 2) is the (x,y) of the flow vector
    flow = np.zeros((h, w, 2), dtype=np.float32)

    # calculate difference per matching pixel from old and new:
    for new, old in zip(good_new, good_old):
        x_new, y_new = int(new[0]), int(new[1])  # [0] is x value, [1] is y value
        x_old, y_old = int(old[0]), int(old[1])  # ||
        flow[y_new, x_new] = [x_new - x_old, y_new - y_old]

    # aggregate result values: magnitude_sum, angular_deviation, magnitude_deviation
    return calculate_flow_metrics(flow)


def calculate_flow_metrics(flow: np.ndarray):
    """
    Calculates magnitude sum, magnitude deviation, and angular deviation of the optical flow vectors.

    Parameters:
        flow (numpy.ndarray): Optical flow with shape (height, width, 2), where the last dimension contains
                              the horizontal (dx) and vertical (dy) flow components.

    Returns:
        dict: A dictionary containing the magnitude sum, magnitude deviation, and angular deviation.
    """
    # Separate the horizontal and vertical components of the flow
    dx, dy = flow[..., 0], flow[..., 1]

    # Calculate the magnitude and angle of the flow vectors
    magnitude, angle = cv2.cartToPolar(dx, dy)

    # Calculate metrics
    magnitude_sum = np.sum(magnitude)  # Total sum of magnitudes
    magnitude_deviation = np.std(magnitude)  # Standard deviation of magnitudes
    angular_deviation = np.std(angle)  # Standard deviation of angles

    return {
        "magnitude_sum": magnitude_sum,
        "magnitude_deviation": magnitude_deviation,
        "angular_deviation": angular_deviation
    }


def extract_variants(frame1: np.ndarray, frame2: np.ndarray, frame_pair_variant_extract_functions: Dict):
    variants = dict()

    for key in frame_pair_variant_extract_functions.keys():
        current_kwargs = frame_pair_variant_extract_functions[key]["kwargs"]
        current_variant_extract_function = frame_pair_variant_extract_functions[key]["function"]

        variants[key] = current_variant_extract_function(frame1, frame2, **current_kwargs)
        
    return variants


"""
input: Video
output: mean velocity, velocity magnitude s-deviation + angular s-deviation, timestamp

main experiment:
input: VideoCapture(), which data to extract from each frame-pair (default = all)
output: append result data to file, display lucas-kanade results over the original video

cv_params = parameters for the CV functions.
hyper_params = super_pixel_shape, ...
def super_function(capture_obj: VideoCapture, cv_params: dict, hyper_params: tuple):
    for result in results:
        output_file.append(result)
    display(results[-1])


def display(result):
    aggregate to super pixels
    display results graph
    display angular s-deviation in a clock
    

benchmark the time overhead of each of the methods.
scale results with superpixel size.

"""
