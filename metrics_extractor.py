#!/usr/bin/env python3
"""
metrics_extractor.py
"""

import csv
import os
import cv2
import numpy as np
from typing import Dict, Any, Tuple
import cv_fish_configuration as conf

def get_std_angle(angles, angular_mean):
    offset = np.pi - angular_mean
    # TODO: add `offset` to all the angles
    # TODO: calculate angular_std from np.pi by numerical value
    # TODO: return angular_std
    pass


def to_gray(frame1: np.ndarray, frame2: np.ndarray, should_apply_gaussian_denoiser: bool=False):
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

    if (should_apply_gaussian_denoiser):
        # denoise the grayscale frames:
        gray1 = cv2.fastNlMeansDenoising(gray1, None, 10, 7, 21)
        gray2 = cv2.fastNlMeansDenoising(gray2, None, 10, 7, 21)

    return gray1, gray2


def extract_farneback_metric(frame1: np.ndarray, frame2: np.ndarray, 
                                pyr_scale: float, levels: int, winsize: int, 
                                iterations: int, poly_n: int, poly_sigma: float,
                                flags: int, should_apply_gaussian_denoiser: bool=False):
    """
    Extracts the farneback metrics from the calculated flow between frame1 and frame2.

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
    gray1, gray2 = to_gray(frame1, frame2, should_apply_gaussian_denoiser)

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

    # aggregate result values: magnitude_mean, angular_deviation, magnitude_deviation
    flow_metrics = calculate_flow_metrics(farneback_flow)
    flow_metrics["flow_matrix"] = farneback_flow

    return flow_metrics


def extract_TVL1_metric(frame1: np.ndarray, frame2: np.ndarray, should_apply_gaussian_denoiser: bool=False):
    """
    Extracts the TVL1 metrics from the calculated flow between frame1 and frame2.

    Parameters:
        frame1 (numpy.ndarray): The first (non-gray scale) frame.
        frame2 (numpy.ndarray): The second (non-gray scale) frame.

    Returns:
        dict: A dictionary containing the magnitude sum, magnitude deviation, and angular deviation
              of the TVL1 flow calculation.
    """
    gray1, gray2 = to_gray(frame1, frame2, should_apply_gaussian_denoiser)

    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = tvl1.calc(gray1, gray2, None)

    # aggregate result values: magnitude_mean, angular_deviation, magnitude_deviation
    return calculate_flow_metrics(flow)


def extract_lucas_kanade_metric(frame1: np.ndarray, frame2: np.ndarray, 
                                    win_size: Tuple[int, int], max_level: int,
                                    criteria: Tuple[int, int, float], max_corners: int,
                                    quality_level: float, min_distance: int, 
                                    block_size: int, should_apply_gaussian_denoiser: bool=False):
    """
    Extracts the lucas kanade metrics from the calculated flow between frame1 and frame2.

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
    gray1, gray2 = to_gray(frame1, frame2, should_apply_gaussian_denoiser)
    
    # init good features to track params
    feature_params = dict()
    feature_params['maxCorners'] = max_corners
    feature_params['qualityLevel'] = quality_level
    feature_params['minDistance'] = min_distance
    feature_params['blockSize'] = block_size

    # init lucas kanade params
    lk_params = dict()
    lk_params['winSize'] = win_size
    lk_params['maxLevel'] = max_level
    lk_params['criteria'] = criteria

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
        if (0 <= y_new < frame1.shape[0] and 
            0 <= x_new < frame1.shape[1]):
            flow[y_new, x_new] = [x_new - x_old, y_new - y_old]

    # aggregate result values: magnitude_mean, angular_deviation, magnitude_deviation
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
    # Separate the horizontal and vertical components of the flow 'W H 2' -> 'W H 1, W H 1'
    dx, dy = flow[..., 0], flow[..., 1]

    x_sum, y_sum = np.array(float(np.sum(dx))), np.array(float(np.sum(dy)))

    # Calculate the magnitude and angle of the flow vectors
    magnitude, angle = cv2.cartToPolar(dx, dy)

    # Calculate metrics
    magnitude_mean = np.mean(magnitude)  # Total sum of magnitudes
    magnitude_deviation = np.std(magnitude)  # Standard deviation of magnitudes
    angular_deviation = np.std(angle)  # Standard deviation of angles
    _, angular_mean = cv2.cartToPolar(x_sum, y_sum)
    angular_mean = angular_mean[0]

    return {
        "magnitude_mean": magnitude_mean,
        "magnitude_deviation": magnitude_deviation,
        "angular_deviation": angular_deviation,
        "angular_mean": angular_mean
    }


def append_metrics(output_path: str, metrics, time_units):
     # Make sure all parent directories exist
    directory = os.path.dirname(output_path)
    if directory:  # Only try to create if there's an actual directory path
        os.makedirs(directory, exist_ok=True)
    
    # If file doesn't exist, we'll need to write the header
    file_exists = os.path.exists(output_path)
    
    with open(output_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header if this is a new file
        if not file_exists:
            writer.writerow(["metric_name",
                             "time", 
                             "magnitude_mean", 
                             "magnitude_deviation", 
                             "angular_deviation", 
                             "angular_mean"])

        for metric_name in metrics.keys():
            # Append the data row
            writer.writerow([
                metric_name,
                time_units,
                metrics[metric_name]['magnitude_mean'],
                metrics[metric_name]['magnitude_deviation'],
                metrics[metric_name]['angular_deviation'],
                metrics[metric_name]['angular_mean']
            ])


def extract_metrics(frame1: np.ndarray, frame2: np.ndarray, metric_extract_functions: Dict):
    """
    Extracting metric metrics from the 2 given frames, running all the metric extract fucntions.

    Parameters:
        frame1 (numpy.ndarray): The 1st frame.
        frame2 (numpy.ndarray): The 2nd frame.
        metric_extract_functions (Dict["metric name", Dict["function", "kwargs"]]):
            a dictionary of the form:
            {
                "metric_name1": {
                    "function": func_name1(frame1, frame2, **kwargs),
                    "kwargs": {
                        "arg1_name": arg1_value,
                        "arg2_name": arg2_value,
                        ...
                    }
                }, 
                "metric_name2": {
                    "function": func_name2(frame1, frame2, **kwargs),
                    "kwargs": {
                        "arg1_name": arg1_value,
                        "arg2_name": arg2_value,
                        ...
                    }
                },
                ...
            }

    Returns:
        dict: A dictionary containing the magnitude sum, 
    """
    metrics = dict()

    for metric_name in metric_extract_functions.keys():
        current_metric_extract_function = metric_extract_functions[metric_name]["function"]
        current_kwargs = metric_extract_functions[metric_name]["kwargs"]

        metrics[metric_name] = current_metric_extract_function(frame1, frame2, **current_kwargs)
        
    return metrics


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
