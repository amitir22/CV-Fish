import sys
import numpy as np
import cv2 as cv
from typing import Tuple
from collections import deque
from datetime import datetime
from time import sleep
from timer import Timer
from metrics_extractor import extract_farneback_metric, extract_TVL1_metric, \
                                extract_lucas_kanade_metric, extract_metrics, append_metrics
import live_bollinger_gui
import cv_fish_configuration as conf


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


def process_frame_window(frames: deque, metric_extractors):
    current_timestamp = datetime.now()

    if len(frames) < 2:
        raise RuntimeError("frame window size must have at least 2 frames.")
    
    frame1 = frames[0]   # take first frame in frame window
    frame2 = frames[-1]  # take last frame in frame window

    metrics = extract_metrics(frame1, frame2, metric_extractors)
    
    return metrics


def main():
    # is_webcam = True
    # # If program was used:
    # # /path> python main.py path/to/file.extension
    # if len(sys.argv) > 1:
    #     is_webcam = False
    is_webcam = False

    if is_webcam:
        video_source = 0
    else:  # video source is a video file
        # video_source = sys.argv[1]
        video_source = './Workable Data/Processed/DPH21_Above_IR10.avi'
        
    video_capture_object = cv.VideoCapture(video_source)

    batch_size = conf.FRAME_WINDOW_SIZE  # Number of frames in the sliding window

    if not video_capture_object.isOpened():
        if is_webcam:
            raise IOError(f"Cannot open webcam.")
        else:
            raise IOError(f"Cannot open video file: {video_source}.")

    timer_object = Timer()

    super_pixel_dimensions = conf.DEFAULT_SUPER_PIXEL_DIMEMSNIONS  # can be modified
    frame_window_size = conf.FRAME_WINDOW_SIZE

    frame_window = deque()  # Init as deque so i can pop left with minimal overhead

    for i in range(frame_window_size):
        ret, frame = get_next_frame(video_capture_object, super_pixel_dimensions)  # Read a frame

        if not ret:
            if is_webcam:
                print("Error: ")
                # added a sleep() here so it won't retry in a short-circuit loop that might
                # cause the webcam driver to cry
                sleep(conf.WEBCAM_RETRY_INTERVAL_SECONDS)
            else:
                print("Error: End of video too soon")
                break

        frame_window.append(frame)

    timer_object.tick()  # Start the clock

    # Setup the GUI
    chart = live_bollinger_gui.MultiBollingerChart(t_window=conf.T_WINDOW, num_std=conf.BOLLINGER_NUM_STD_OF_BANDS)

    try:
        frame_count = frame_window_size    
        now = timer_object.start_time
        output_prefix = f'{now.year}-{now.month:02d}-{now.day:02d}-{now.hour:02d}-{now.minute:02d}'

        if is_webcam:
            output_suffix = 'time_sec'
        else:  # is video file
            output_suffix = 'frame_count'

        # prepare the metric extractors
        metric_extractors = dict()
        metric_extractors["Lucas-Kanade"] = {
            "kwargs": conf.DEFAULT_LUCAS_KANADE_PARAMS,
            "function": extract_lucas_kanade_metric
        }
        metric_extractors["Farneback"] = {
            "kwargs": conf.DEFAULT_FARNEBACK_PARAMS,
            "function": extract_farneback_metric
        }
        metric_extractors["TVL1"] = {
            "kwargs": conf.DEFAULT_TVL1_PARAMS,
            "function": extract_TVL1_metric
        }
        # TODO: add more metric extractors here. (function: Func[frame1, frame2, params/kwargs...] => Dict)

        while True:
            ret, frame = get_next_frame(video_capture_object, super_pixel_dimensions)  # Read a frame

            if not ret:
                if is_webcam:
                    print("Error: ")
                    # added a sleep() here so it won't retry in a short-circuit loop that might
                    # cause the webcam driver to cry :(
                    sleep(conf.WEBCAM_RETRY_INTERVAL_SECONDS)
                else: # video source is a file
                    print("End of video")
                    break

            frame_window.popleft()
            frame_window.append(frame)

            if is_webcam:  # t-axis is elapsed seconds
                # taking the elapsed before, so i can know the time of when the last frame was taken.
                elapsed_seconds = timer_object.tock() // 1_000_000  # convert microsec to sec
                elapsed_t_units = elapsed_seconds
            else:  # If file then the t-axis is frame count
                elapsed_t_units = frame_count

            # NOTE: this is the HEAVY function
            # metrics is now a dictionary of the form: {"varient_name": <metrics dictionary>}
            metrics = process_frame_window(frame_window, metric_extractors)

            # get flow matrix from farneback separately
            flow = metrics["Farneback"]["flow_matrix"]

            # log the metrics with the elapsed units
            output_file_path = f'./output/{output_prefix}_{video_source.split("/")[-1][:15]}_flow_metrics_{output_suffix}.csv'
            append_metrics(output_file_path, metrics, elapsed_t_units)

            # display updating video and graph
            data_dict = {}

            # NOTE: decide which data to provide to the Bollinger chart

            # # Display all metrics on the bollinger chart:
            # for line in metric_extractors.keys():
            #     data_dict[f'{line}_magnitude_mean'] = (metrics["Farneback"]['magnitude_mean'], metrics["Farneback"]['magnitude_deviation'])
            #     data_dict[f'{line}_angular_mean'] = (metrics["Farneback"]['angular_mean'], metrics["Farneback"]['angular_deviation'])

            # Display only the Farneback metrics to the bollinger chart:
            data_dict["Farneback_magnitude_mean"] = (metrics["Farneback"]['magnitude_mean'], metrics["Farneback"]['magnitude_deviation'])

            # Note: Doesn't work yet:
            # data_dict["Farneback_angular_mean"] = (metrics["Farneback"]['angular_mean'], metrics["Farneback"]['angular_deviation'])
            
            chart.push_new_data(data_dict, frame_window[0], flow)

            frame_count += 1

    finally:  # release resources
        video_capture_object.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    main()
