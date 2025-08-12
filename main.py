import sys
import numpy as np
import cv2 as cv
from typing import Tuple
from datetime import datetime
from time import sleep
from metrics_extractor import extract_farneback_metric, extract_TVL1_metric, \
                                extract_lucas_kanade_metric, extract_metrics, append_metrics
import live_bollinger_gui
import cv_fish_configuration as conf

VIDOE_FILE_PATH = './Workable Data/Processed/DPH21_Above_IR10.avi'

# TODO: fill the correct data
NVR_USER = 'admin'
NVR_PASS = 'admin12345'
NVR_IP = '0.0.0.0'  # within network
NVR_PORT = '554'  # default port for the protocol, might not need change
NVR_PATH = '/Streaming/Channels/101'

VIDEO_SOURCE = {
    'FILE': VIDOE_FILE_PATH,
    'WEBCAM': 0,
    'NVR': f"rtsp://{NVR_USER}:{NVR_PASS}@{NVR_IP}:{NVR_PORT}{NVR_PATH}"
}


def get_next_frame(video_capture_object: cv.VideoCapture, 
                   super_pixel_shape: Tuple[int, int]=conf.DEFAULT_SUPER_PIXEL_SHAPE):
    """
    Getting the next frame, performing binning (applying super pixel here for enhanced performance).
    """
    ret, frame = video_capture_object.read()

    # apply super pixel preprocessing when retrieving frame
    if ret:
        frame = cv.resize(frame, (frame.shape[1] // super_pixel_shape[1], 
                                  frame.shape[0] // super_pixel_shape[0]),
                                  interpolation=cv.INTER_LINEAR)

    return ret, frame


def main():
    # is_webcam = True
    # # If program was used:
    # # /path> python main.py path/to/file.extension
    # if len(sys.argv) > 1:
    #     is_webcam = False
    is_webcam = False
    is_nvr = True

    if is_webcam:
        video_source = VIDEO_SOURCE["WEBCAM"]
    elif is_nvr:
        video_source = VIDEO_SOURCE["NVR"]
    else:  # video source is a video file
        # video_source = sys.argv[1]
        video_source = VIDEO_SOURCE["FILE"]
        
    super_pixel_dimensions = conf.DEFAULT_SUPER_PIXEL_DIMEMSNIONS
    frame_window_size = conf.FRAME_WINDOW_SIZE

    chart = live_bollinger_gui.MultiBollingerChart(
        t_window=conf.T_WINDOW, num_std=conf.BOLLINGER_NUM_STD_OF_BANDS
    )

    # prepare the metric extractors
    metric_extractors = {
        "Lucas-Kanade": {
            "kwargs": conf.DEFAULT_LUCAS_KANADE_PARAMS,
            "function": extract_lucas_kanade_metric,
        },
        "Farneback": {
            "kwargs": conf.DEFAULT_FARNEBACK_PARAMS,
            "function": extract_farneback_metric,
        },
        "TVL1": {"kwargs": conf.DEFAULT_TVL1_PARAMS, "function": extract_TVL1_metric},
    }

    try:
        while True:
            video_capture_object = cv.VideoCapture(video_source)

            if not video_capture_object.isOpened():
                if is_webcam:
                    raise IOError("Cannot open webcam.")
                else:
                    raise IOError(f"Cannot open video source: {video_source}.")

            frame_window = []
            for _ in range(frame_window_size):
                ret, frame = get_next_frame(video_capture_object, super_pixel_dimensions)
                if not ret:
                    break
                frame_window.append(frame)

            video_capture_object.release()

            if len(frame_window) < frame_window_size:
                print("Unable to capture enough frames")
                break

            timestamp = datetime.now().isoformat()
            date_stamp = datetime.now().strftime("%Y%m%d")
            output_file_path = f"./output/{date_stamp}.csv"

            pair_metrics = {}
            for idx in range(1, frame_window_size):
                pair_name = f"1-{idx+1}"
                metrics = extract_metrics(frame_window[0], frame_window[idx], metric_extractors)
                append_metrics(output_file_path, metrics, timestamp, pair_name)
                pair_metrics[pair_name] = metrics

            data_dict = {}
            for pair_name, metrics in pair_metrics.items():
                fb = metrics["Farneback"]
                data_dict[f"{pair_name}:Farneback_magnitude_mean"] = (
                    fb["magnitude_mean"],
                    fb["magnitude_deviation"],
                )
            flow = pair_metrics["1-2"]["Farneback"]["flow_matrix"]
            chart.push_new_data(data_dict, frame_window[0], flow)

            sleep(conf.CAPTURE_INTERVAL_MINUTES * 60)

    finally:
        cv.destroyAllWindows()


if __name__ == '__main__':
    main()
