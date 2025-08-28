#!/usr/bin/env python3
"""Background thread that captures frames and updates metric CSV/images."""

from __future__ import annotations

import os
import shutil
import threading
import time
from datetime import datetime
from typing import Tuple, Any

import cv2 as cv
import requests

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


def _capture_sample(video_source: str,
                    num_frames: int,
                    super_pixel_dimensions: Tuple[int, int]):
    """Capture a fixed number of frames with retry on failure."""
    for attempt in range(conf.CAPTURE_RETRY_ATTEMPTS):
        cap = cv.VideoCapture(video_source)
        frames = []
        for _ in range(num_frames):
            ret, frame = _get_next_frame(cap, super_pixel_dimensions)
            if not ret:
                break
            frames.append(frame)
        cap.release()
        if len(frames) == num_frames:
            return frames
        time.sleep(conf.WEBCAM_RETRY_INTERVAL_SECONDS)
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


def _post_log(level: str, message: str, timestamp: str) -> None:
    """Send a log entry to the Flask server; failures are ignored."""
    try:
        requests.post(
            conf.LOG_ENDPOINT,
            json={'timestamp': timestamp, 'level': level, 'message': message},
            timeout=5,
        )
    except Exception:
        pass


def _prepare_output_paths():
    """Ensure directories and placeholder files exist for metrics output."""
    # Create output and frame directories
    os.makedirs(conf.FRAMES_DIR, exist_ok=True)
    os.makedirs(conf.OUTPUT_DIR, exist_ok=True)
    # Touch latest-frame paths so later writes don't fail
    for path in (conf.LATEST_FRAME_PATH, conf.LATEST_TS_PATH):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if not os.path.exists(path):
            open(path, 'a', encoding='utf-8').close()


def _metrics_loop(stop_event: threading.Event, runtime_cfg: dict[str, Any], cfg_lock: threading.Lock):
    """Continuously capture frames and update CSV/image outputs."""
    _prepare_output_paths()

    is_webcam = False
    is_nvr = True

    if is_webcam:
        video_source = conf.VIDEO_SOURCE['WEBCAM']
    elif is_nvr:
        video_source = conf.VIDEO_SOURCE['NVR']
    else:
        video_source = conf.VIDEO_SOURCE['FILE']

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
        with cfg_lock:
            frame_window_size = runtime_cfg.get('FRAME_WINDOW_SIZE', conf.FRAME_WINDOW_SIZE)
            capture_interval_minutes = runtime_cfg.get('CAPTURE_INTERVAL_MINUTES', conf.CAPTURE_INTERVAL_MINUTES)
            super_pixel_dimensions = runtime_cfg.get('DEFAULT_SUPER_PIXEL_DIMEMSNIONS', conf.DEFAULT_SUPER_PIXEL_DIMEMSNIONS)
        capture_interval = capture_interval_minutes * 60

        try:
            frames = _capture_sample(video_source, frame_window_size, super_pixel_dimensions)
            now = datetime.now()
            time_stamp = now.strftime(conf.TIMESTAMP_FORMAT)
            if len(frames) < frame_window_size:
                _post_log('error', 'insufficient frames captured', time_stamp)
                stop_event.wait(capture_interval)
                continue

            date_str = now.strftime('%Y%m%d')
            output_file_path = os.path.join(conf.OUTPUT_DIR, f'{date_str}.csv')

            flows = []
            for idx in range(1, frame_window_size):
                metrics = extract_metrics(frames[0], frames[idx], metric_extractors)
                pair_label = f'1-{idx+1}'
                append_metrics(output_file_path, metrics, time_stamp, pair_label)
                flows.append(metrics['Farneback'].get('flow_matrix'))

            _save_frame_window(frames, flows, time_stamp)
            _post_log('success', 'frame window processed', time_stamp)
        except Exception as exc:
            err_ts = datetime.now().strftime(conf.TIMESTAMP_FORMAT)
            _post_log('error', str(exc), err_ts)

        stop_event.wait(capture_interval)


def start_metrics_thread(stop_event: threading.Event,
                        runtime_cfg: dict[str, Any],
                        cfg_lock: threading.Lock) -> threading.Thread:
    """Start the metrics extraction thread."""
    thread = threading.Thread(
        target=_metrics_loop,
        args=(stop_event, runtime_cfg, cfg_lock),
        daemon=True,
    )
    thread.start()
    return thread

