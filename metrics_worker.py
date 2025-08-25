#!/usr/bin/env python3
"""Background thread that captures frames and updates metric CSV/images."""

from __future__ import annotations

import os
import shutil
import threading
from datetime import datetime
from typing import Tuple

import cv2 as cv

from metrics_extractor import (
    extract_farneback_metric,
    extract_TVL1_metric,
    extract_lucas_kanade_metric,
    extract_metrics,
    append_metrics,
)
import cv_fish_configuration as conf


def _get_next_frame(video_capture_object: cv.VideoCapture,
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


def _capture_sample(video_source: str, num_frames: int, super_pixel_dimensions: Tuple[int, int]):
    """Capture a fixed number of frames from the given video source."""
    cap = cv.VideoCapture(video_source)
    frames = []
    for _ in range(num_frames):
        ret, frame = _get_next_frame(cap, super_pixel_dimensions)
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def _draw_flow_quivers(frame, flow, step: int = 60):
    """Overlay a sparse quiver plot of optical flow vectors on the frame."""
    if flow is None:
        return frame
    h, w = frame.shape[:2]
    vis = frame.copy()
    for y in range(0, h, step):
        for x in range(0, w, step):
            fx, fy = flow[y, x]
            end_point = (int(x + fx), int(y + fy))
            cv.arrowedLine(vis, (x, y), end_point, (0, 0, 255), 1, tipLength=0.3)
    return vis


def _save_frame_window(frames, flows, timestamp: str):
    """Save all frames with quiver overlays and update the latest pointers."""
    os.makedirs(conf.FRAMES_DIR, exist_ok=True)
    last_path = None
    # Save the first frame without quivers
    base_path = os.path.join(conf.FRAMES_DIR, f'frame1-{timestamp}.png')
    cv.imwrite(base_path, frames[0])
    last_path = base_path
    for idx in range(1, len(frames)):
        flow = flows[idx - 1]
        vis = _draw_flow_quivers(frames[idx], flow)
        path = os.path.join(conf.FRAMES_DIR, f'frame{idx + 1}-{timestamp}.png')
        cv.imwrite(path, vis)
        last_path = path
    if last_path:
        shutil.copy(last_path, conf.LATEST_FRAME_PATH)
        with open(conf.LATEST_TS_PATH, 'w', encoding='utf-8') as fh:
            fh.write(timestamp)


def _metrics_loop(stop_event: threading.Event):
    """Continuously capture frames and update CSV/image outputs."""
    is_webcam = False
    is_nvr = True

    if is_webcam:
        video_source = conf.VIDEO_SOURCE['WEBCAM']
    elif is_nvr:
        video_source = conf.VIDEO_SOURCE['NVR']
    else:
        video_source = conf.VIDEO_SOURCE['FILE']

    super_pixel_dimensions = conf.DEFAULT_SUPER_PIXEL_DIMEMSNIONS
    capture_interval = conf.CAPTURE_INTERVAL_MINUTES * 60

    metric_extractors = {
        'Lucas-Kanade': {
            'kwargs': conf.DEFAULT_LUCAS_KANADE_PARAMS,
            'function': extract_lucas_kanade_metric
        },
        'Farneback': {
            'kwargs': conf.DEFAULT_FARNEBACK_PARAMS,
            'function': extract_farneback_metric
        },
        'TVL1': {
            'kwargs': conf.DEFAULT_TVL1_PARAMS,
            'function': extract_TVL1_metric
        },
    }

    while not stop_event.is_set():
        frames = _capture_sample(video_source, conf.FRAME_WINDOW_SIZE, super_pixel_dimensions)
        if len(frames) < conf.FRAME_WINDOW_SIZE:
            stop_event.wait(capture_interval)
            continue

        now = datetime.now()
        date_str = now.strftime('%Y%m%d')
        output_file_path = os.path.join(conf.OUTPUT_DIR, f'{date_str}.csv')
        time_stamp = now.strftime('%Y%m%d-%H%M')

        flows = []
        for idx in range(1, conf.FRAME_WINDOW_SIZE):
            metrics = extract_metrics(frames[0], frames[idx], metric_extractors)
            pair_label = f'1-{idx+1}'
            append_metrics(output_file_path, metrics, time_stamp, pair_label)
            flows.append(metrics['Farneback'].get('flow_matrix'))

        _save_frame_window(frames, flows, time_stamp)

        stop_event.wait(capture_interval)


def start_metrics_thread(stop_event: threading.Event) -> threading.Thread:
    """Start the metrics extraction thread."""
    thread = threading.Thread(target=_metrics_loop, args=(stop_event,), daemon=True)
    thread.start()
    return thread

