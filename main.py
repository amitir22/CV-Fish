import numpy as np
import cv2 as cv
from typing import Tuple
from collections import deque
from datetime import datetime
from time import sleep
from timer import Timer
from variants_extractor import DEFAULT_FARNEBACK_PARAMS, DEFAULT_LUCAS_KANADE_PARAMS, DEFAULT_TVL1_PARAMS, \
extract_farneback_variant, extract_TVL1_variant, extract_lucas_kanade_variant, extract_variants, append_metrics
import live_bollinger_gui


T_WINDOW = 100
BOLLINGER_NUM_STD_OF_BANDS = 2.0
DEFAULT_SUPER_PIXEL_SHAPE = (1, 1)


def get_next_frame(video_capture_object: cv.VideoCapture, 
                   super_pixel_shape: Tuple[int, int]=DEFAULT_SUPER_PIXEL_SHAPE):
    ret, frame = video_capture_object.read()

    # apply super pixel preprocessing when retrieving frame
    if ret:
        frame = cv.resize(frame, (frame.shape[1] // super_pixel_shape[1], 
                                  frame.shape[0] // super_pixel_shape[0]),
                                  interpolation=cv.INTER_LINEAR)

    return ret, frame


def process_frame_window(frames: deque, variant_extractors):
    current_timestamp = datetime.now()

    if len(frames) < 2:
        raise RuntimeError("frame window size must have at least 2 frames.")
    
    frame1 = frames[0]   # take first frame in frame window
    frame2 = frames[-1]  # take last frame in frame window

    variants = extract_variants(frame1, frame2, variant_extractors)
    
    return variants


def main():
    # TODO: toggle: 
    #           True: capture from webcam. 
    #           False: capture from file.
    IS_WEBCAM = False

    if IS_WEBCAM:
        video_source = 0
    else:  # video source is a video file
        video_source = "Workable Data/Processed/DPH0_Surface_TL.avi"
        
    video_capture_object = cv.VideoCapture(video_source)

    batch_size = 5  # Number of frames in the sliding window

    if not video_capture_object.isOpened():
        if IS_WEBCAM:
            raise IOError(f"Cannot open webcam.")
        else:
            raise IOError(f"Cannot open video file: {video_source}.")

    timer_object = Timer()

    super_pixel_dimensions = (4, 4)  # can be modified
    frame_window_size = 5

    frame_window = deque()  # Init as deque so i can pop left with minimal overhead

    for i in range(frame_window_size):
        ret, frame = get_next_frame(video_capture_object, super_pixel_dimensions)  # Read a frame

        if not ret:
            if IS_WEBCAM:
                print("Error: ")
                # added a sleep(2) here so it won't retry in a short-circuit loop that might
                # cause the webcam driver to cry
                sleep(2)
            else:
                print("Error: End of video too soon")
                break

        frame_window.append(frame)

    timer_object.tick()  # Start the clock

    # Setup the GUI
    chart = live_bollinger_gui.MultiBollingerChart(t_window=T_WINDOW, num_std=BOLLINGER_NUM_STD_OF_BANDS)

    try:
        frame_count = frame_window_size    
        now = timer_object.start_time
        output_prefix = f'{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}'

        if IS_WEBCAM:
            output_suffix = 'time_sec'
        else:  # is video file
            output_suffix = 'frame_count'

        # prepare the variant extractors
        variant_extractors = dict()
        variant_extractors["Lucas-Kanade"] = {
            "kwargs": DEFAULT_LUCAS_KANADE_PARAMS,
            "function": extract_lucas_kanade_variant
        }
        variant_extractors["Farneback"] = {
            "kwargs": DEFAULT_FARNEBACK_PARAMS,
            "function": extract_farneback_variant
        }
        variant_extractors["TVL1"] = {
            "kwargs": DEFAULT_TVL1_PARAMS,
            "function": extract_TVL1_variant
        }
        # TODO: add more variant extractors here

        while True:
            ret, frame = get_next_frame(video_capture_object, super_pixel_dimensions)  # Read a frame

            if not ret:
                if IS_WEBCAM:
                    print("Error: ")
                    # TODO: maybe add a sleep(2) here so it won't retry in a short-circuit loop
                else: # video source is a file
                    print("End of video")
                    break

            frame_window.popleft()
            frame_window.append(frame)

            if IS_WEBCAM:  # t-axis is elapsed seconds
                # taking the elapsed before, so i can know the time of when the last frame was taken.
                elapsed_seconds = timer_object.tock() // 1_000_000  # convert microsec to sec
                elapsed_t_units = elapsed_seconds
            else:  # If file then the t-axis is frame count
                elapsed_t_units = frame_count

            # NOTE: this is the HEAVY function
            # metrics is now a dictionary of the form: {"varient_name": <metrics dictionary>}
            metrics = process_frame_window(frame_window, variant_extractors)

            # log the metrics with the elapsed units
            output_file_path = f'./output/{output_prefix}_{video_source.split("/")[-1][:15]}_flow_metrics_{output_suffix}.csv'
            append_metrics(output_file_path, metrics, elapsed_t_units)

            # display updating video and graph
            data_dict = {}

            for line in variant_extractors.keys():
                data_dict[line] = (metrics[line]['magnitude_mean'], metrics[line]['magnitude_deviation'])
            
            chart.push_new_data(data_dict, frame_window[0])

            frame_count += 1

    finally:  # release resources
        video_capture_object.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    main()
