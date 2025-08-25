import sys
"""Capture frames, compute optical-flow metrics and update the UI.

This module orchestrates the full data-processing loop: it repeatedly
captures a short window of frames from a configured video source,
computes multiple optical-flow metrics for several frame pairs and logs
the results to CSV while feeding the live Bollinger chart.
"""

import os
import numpy as np
import cv2 as cv
from typing import Tuple
from datetime import datetime
from metrics_extractor import (
    extract_farneback_metric,
    extract_TVL1_metric,
    extract_lucas_kanade_metric,
    extract_metrics,
    append_metrics,
)
import live_bollinger_gui
import cv_fish_configuration as conf


def get_next_frame(video_capture_object: cv.VideoCapture,
                   super_pixel_shape: Tuple[int, int] = conf.DEFAULT_SUPER_PIXEL_SHAPE):
    """Get the next frame and apply super pixel downscaling."""
    ret, frame = video_capture_object.read()
    if ret:
        frame = cv.resize(
            frame,
            (frame.shape[1] // super_pixel_shape[1], frame.shape[0] // super_pixel_shape[0]),
            interpolation=cv.INTER_LINEAR,
        )
    return ret, frame


def capture_sample(video_source: str, num_frames: int, super_pixel_dimensions: Tuple[int, int]):
    """Capture a fixed number of frames from the given video source."""
    cap = cv.VideoCapture(video_source)
    frames = []
    for _ in range(num_frames):
        ret, frame = get_next_frame(cap, super_pixel_dimensions)
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def main():
    """Continuously capture frame windows and process optical-flow metrics."""
    # Determine which video source to use.  For now we favour the NVR
    # stream but the logic is easily adaptable for other sources.
    # is_webcam = True
    # if len(sys.argv) > 1:
    #     is_webcam = False
    is_webcam = False
    is_nvr = True

    if is_webcam:
        video_source = conf.VIDEO_SOURCE["WEBCAM"]
    elif is_nvr:
        video_source = conf.VIDEO_SOURCE["NVR"]
    else:  # video source is a video file
        # video_source = sys.argv[1]
        video_source = conf.VIDEO_SOURCE["FILE"]

    super_pixel_dimensions = conf.DEFAULT_SUPER_PIXEL_DIMEMSNIONS  # can be modified

    # Prepare metric extractors once
    metric_extractors = {
        "Lucas-Kanade": {"kwargs": conf.DEFAULT_LUCAS_KANADE_PARAMS, "function": extract_lucas_kanade_metric},
        "Farneback": {"kwargs": conf.DEFAULT_FARNEBACK_PARAMS, "function": extract_farneback_metric},
        "TVL1": {"kwargs": conf.DEFAULT_TVL1_PARAMS, "function": extract_TVL1_metric},
    }

    # Prepare the live chart UI and pair labels for frame comparisons
    pair_labels = [f"1-{i}" for i in range(2, conf.FRAME_WINDOW_SIZE + 1)]
    chart = live_bollinger_gui.MultiPairBollingerChart(
        pair_labels, t_window=conf.T_WINDOW, num_std=conf.BOLLINGER_NUM_STD_OF_BANDS
    )

    capture_interval = conf.CAPTURE_INTERVAL_MINUTES * 60

    while True:
        # Capture a short sequence of frames from the source
        frames = capture_sample(video_source, conf.FRAME_WINDOW_SIZE, super_pixel_dimensions)
        if len(frames) < conf.FRAME_WINDOW_SIZE:
            print("Failed to capture enough frames")
            chart.wait_with_ui(capture_interval)
            continue

        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        output_file_path = os.path.join(conf.OUTPUT_DIR, f"{date_str}.csv")
        time_stamp = now.isoformat()

        # Compare the first frame with every subsequent frame in the window
        for idx in range(1, conf.FRAME_WINDOW_SIZE):
            metrics = extract_metrics(frames[0], frames[idx], metric_extractors)
            pair_label = f"1-{idx+1}"
            append_metrics(output_file_path, metrics, time_stamp, pair_label)

            data_dict = {
                algo: (
                    metrics[algo]["magnitude_mean"],
                    metrics[algo]["magnitude_deviation"],
                )
                for algo in metric_extractors.keys()
            }
            flow = metrics["Farneback"].get("flow_matrix")
            chart.push_new_data(data_dict, frame=frames[0], flow=flow, pair_name=pair_label)

        # Wait before sampling again but keep the UI responsive
        chart.wait_with_ui(capture_interval)

    # Should never reach here but ensure clean up
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
