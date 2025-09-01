#!/usr/bin/env python3
"""Run the metrics extraction thread and expose a Flask REST API."""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import threading
import tempfile
from collections import deque
from typing import Any

import cv2 as cv
from flask import Flask, jsonify, make_response, render_template, send_file, request

import cv_fish_configuration as conf
from metrics_extractor import (
    extract_metrics,
    extract_farneback_metric,
    extract_TVL1_metric,
    extract_lucas_kanade_metric,
)
from metrics_worker import start_metrics_thread

app = Flask(__name__)

# In-memory log storage for worker reports
LOGS_FILE = os.path.join(conf.OUTPUT_DIR, 'worker_logs.json')

# Runtime-configurable settings including nested dictionaries


def _deep_copy(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _deep_copy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_deep_copy(v) for v in obj)
    return obj


def _type_map(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _type_map(v) for k, v in obj.items()}
    return type(obj)


_runtime_config: dict[str, Any] = {
    k: _deep_copy(getattr(conf, k))
    for k in dir(conf)
    if k.isupper()
}
_config_types = {k: _type_map(v) for k, v in _runtime_config.items()}
_config_lock = threading.Lock()


def _load_logs() -> list[dict]:
    if os.path.exists(LOGS_FILE):
        with open(LOGS_FILE, encoding='utf-8') as fh:
            try:
                return json.load(fh)
            except Exception:
                return []
    return []


def _save_logs() -> None:
    os.makedirs(os.path.dirname(LOGS_FILE), exist_ok=True)
    with open(LOGS_FILE, 'w', encoding='utf-8') as fh:
        json.dump(_logs, fh)


_logs: list[dict] = _load_logs()
_logs_lock = threading.Lock()


@app.get('/config')
def get_config():
    """Expose current runtime configuration."""
    with _config_lock:
        return jsonify({'config': _runtime_config})


@app.post('/config')
def update_config():
    """Update runtime configuration values."""
    data = request.get_json(force=True) or {}

    def _apply(cfg: dict, types: dict, updates: dict) -> None:
        for k, v in updates.items():
            if k not in cfg:
                continue
            if isinstance(cfg[k], dict) and isinstance(v, dict):
                _apply(cfg[k], types.get(k, {}), v)
            else:
                typ = types.get(k, type(cfg[k]))
                try:
                    cfg[k] = typ(v)
                except Exception:
                    pass

    with _config_lock:
        _apply(_runtime_config, _config_types, data)
    return ('', 204)


@app.route('/')
def index():
    """Serve the web UI."""
    return render_template('index.html')


@app.get('/metrics')
def get_all_metrics():
    """Return all metrics across CSV files optionally filtered by time."""
    start = request.args.get('start')
    end = request.args.get('end')
    files = sorted(glob.glob(os.path.join(conf.OUTPUT_DIR, '*.csv')))
    rows: list[dict] = []
    for path in files:
        with open(path, newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                ts = row.get('time', '')
                if start and ts < start:
                    continue
                if end and ts > end:
                    continue
                rows.append(row)
    ts = rows[-1]['time'] if rows else None
    return jsonify({'timestamp': ts, 'metrics': rows})


def _process_video(path: str) -> list[dict]:
    """Analyse a video file in one pass and return metric rows."""
    with _config_lock:
        frame_window = _runtime_config.get('FRAME_WINDOW_SIZE', conf.FRAME_WINDOW_SIZE)
        sp = _runtime_config.get('DEFAULT_SUPER_PIXEL_DIMEMSNIONS', conf.DEFAULT_SUPER_PIXEL_DIMEMSNIONS)

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

    cap = cv.VideoCapture(path)
    frames: deque = deque(maxlen=frame_window)
    rows: list[dict] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.resize(
            frame,
            (frame.shape[1] // sp[1], frame.shape[0] // sp[0]),
            interpolation=cv.INTER_LINEAR,
        )
        frames.append(frame)
        if len(frames) < frame_window:
            continue
        ts = f"file-{idx:06d}"
        for j in range(1, frame_window):
            metrics = extract_metrics(frames[0], frames[j], metric_extractors)
            for name, vals in metrics.items():
                rows.append({
                    'pair': f'1-{j+1}',
                    'metric_name': name,
                    'time': ts,
                    'magnitude_mean': vals['magnitude_mean'],
                    'magnitude_deviation': vals['magnitude_deviation'],
                    'angular_deviation': vals['angular_deviation'],
                    'angular_mean': vals['angular_mean'],
                })
        idx += 1
    cap.release()

    out_dir = os.path.join(conf.OUTPUT_DIR, 'video_analysis')
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    csv_path = os.path.join(out_dir, f'{base}.csv')
    if rows:
        with open(csv_path, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow([
                'pair', 'metric_name', 'time', 'magnitude_mean',
                'magnitude_deviation', 'angular_deviation', 'angular_mean'
            ])
            for r in rows:
                writer.writerow([
                    r['pair'], r['metric_name'], r['time'],
                    r['magnitude_mean'], r['magnitude_deviation'],
                    r['angular_deviation'], r['angular_mean']
                ])
    return rows


@app.post('/analyze')
def analyze_video():
    """Accept a video file upload and return analysed metrics."""
    if 'video' not in request.files:
        return ('', 400)
    file = request.files['video']
    if not file.filename:
        return ('', 400)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    try:
        file.save(tmp.name)
        rows = _process_video(tmp.name)
    finally:
        tmp.close()
        os.unlink(tmp.name)
    return jsonify({'metrics': rows})


@app.get('/favicon.ico')
def favicon():
    """Avoid unnecessary 404 errors for the browser favicon request."""
    return ('', 204)


@app.get('/frame')
@app.get('/frame/<timestamp>')
def get_frame(timestamp: str | None = None):
    """Return a frame by timestamp or the most recent one if none given."""
    qs_ts = request.args.get('timestamp')
    if qs_ts and re.fullmatch(r"\d{8}-\d{6}", qs_ts):
        timestamp = qs_ts
    elif timestamp and not re.fullmatch(r"\d{8}-\d{6}", timestamp):
        timestamp = None
    if timestamp:
        pattern = os.path.join(conf.FRAMES_DIR, f'frame*-{timestamp}.png')
        files = sorted(glob.glob(pattern))
        if not files:
            return ('', 404)
        path = files[-1]
    else:
        if not os.path.exists(conf.LATEST_FRAME_PATH):
            return ('', 204)
        path = conf.LATEST_FRAME_PATH
        timestamp = ''
        if os.path.exists(conf.LATEST_TS_PATH):
            with open(conf.LATEST_TS_PATH, encoding='utf-8') as fh:
                timestamp = fh.read().strip()
    response = make_response(send_file(path, mimetype='image/png'))
    if timestamp:
        response.headers['X-Data-Timestamp'] = timestamp
    return response


@app.get('/frames')
def list_frame_sets():
    """Return all timestamps that have saved frame windows."""
    pattern = os.path.join(conf.FRAMES_DIR, 'frame*-*.png')
    files = glob.glob(pattern)
    timestamps = sorted({os.path.basename(p).split('-', 1)[1].rsplit('.', 1)[0] for p in files})
    return jsonify({'timestamps': timestamps})


@app.get('/frames/all')
def list_all_frames():
    """Return all frame filenames grouped by their capture timestamp."""
    pattern = os.path.join(conf.FRAMES_DIR, 'frame*-*.png')
    files = sorted(glob.glob(pattern))
    grouped: dict[str, list[dict[str, int | str]]] = {}
    for path in files:
        base = os.path.basename(path)
        idx_part, ts_part = base.split('-', 1)
        ts = ts_part.rsplit('.', 1)[0]
        idx = int(idx_part.replace('frame', ''))
        grouped.setdefault(ts, []).append({'index': idx, 'filename': base})
    frame_sets = [
        {'timestamp': ts, 'frames': sorted(frames, key=lambda f: f['index'])}
        for ts, frames in sorted(grouped.items())
    ]
    return jsonify({'frame_sets': frame_sets})


@app.get('/frames/<timestamp>')
def list_frames(timestamp: str):
    """List frame indices available for a given capture timestamp."""
    pattern = os.path.join(conf.FRAMES_DIR, f'frame*-{timestamp}.png')
    files = sorted(glob.glob(pattern))
    frames = []
    for p in files:
        base = os.path.basename(p)
        idx = int(base.split('-', 1)[0].replace('frame', ''))
        frames.append({'index': idx, 'filename': base})
    return jsonify({'timestamp': timestamp, 'frames': frames})


@app.get('/frames/<timestamp>/<int:index>')
def get_frame_by_index(timestamp: str, index: int):
    """Retrieve a specific frame image by timestamp and index."""
    path = os.path.join(conf.FRAMES_DIR, f'frame{index}-{timestamp}.png')
    if not os.path.exists(path):
        return ('', 404)
    response = make_response(send_file(path, mimetype='image/png'))
    response.headers['X-Data-Timestamp'] = timestamp
    return response


@app.post('/logs')
def post_log():
    """Receive a log entry from the metrics worker."""
    data = request.get_json(force=True) or {}
    entry = {
        'timestamp': data.get('timestamp'),
        'level': data.get('level', 'info'),
        'message': data.get('message', '')
    }
    with _logs_lock:
        _logs.append(entry)
        _save_logs()
    return ('', 204)


@app.get('/logs')
def get_logs():
    """Return worker log entries sorted by timestamp descending."""
    start = request.args.get('start')
    end = request.args.get('end')
    with _logs_lock:
        filtered = [
            l for l in _logs
            if (not start or l.get('timestamp', '') >= start)
            and (not end or l.get('timestamp', '') <= end)
        ]
        ordered = sorted(filtered, key=lambda x: x.get('timestamp', ''), reverse=True)
    return jsonify({'logs': ordered})


def main():
    """Start the metrics thread and run the Flask development server."""
    # Log the RTSP source used for frame capture so operators know
    # exactly which stream is being consumed.
    rtsp_path = conf.VIDEO_SOURCE['NVR']
    print(f"Capturing frames from RTSP source: {rtsp_path}")

    stop_event = threading.Event()
    thread = start_metrics_thread(stop_event, _runtime_config, _config_lock)
    try:
        app.run(host=conf.FLASK_HOST, port=conf.FLASK_PORT)
    finally:
        stop_event.set()
        thread.join()


if __name__ == '__main__':
    main()
