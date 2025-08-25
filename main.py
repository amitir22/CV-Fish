#!/usr/bin/env python3
"""Run the metrics extraction thread and expose a Flask REST API."""

from __future__ import annotations

import csv
import glob
import os
import threading
from flask import Flask, jsonify, make_response, render_template, send_file

import cv_fish_configuration as conf
from metrics_worker import start_metrics_thread

app = Flask(__name__)


def _latest_csv() -> str | None:
    files = sorted(glob.glob(os.path.join(conf.OUTPUT_DIR, '*.csv')))
    return files[-1] if files else None


@app.route('/')
def index():
    """Serve the web UI."""
    return render_template('index.html')


@app.get('/metrics')
def get_all_metrics():
    """Return all metrics from the latest CSV file."""
    csv_path = _latest_csv()
    if csv_path is None:
        return jsonify({'timestamp': None, 'metrics': []})
    with open(csv_path, newline='') as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    ts = rows[-1]['time'] if rows else None
    return jsonify({'timestamp': ts, 'metrics': rows})


@app.get('/favicon.ico')
def favicon():
    """Avoid unnecessary 404 errors for the browser favicon request."""
    return ('', 204)


@app.get('/frame')
def get_last_frame():
    """Return the most recently saved frame with optical-flow quivers."""
    if not os.path.exists(conf.LATEST_FRAME_PATH):
        return ('', 204)
    ts = ''
    if os.path.exists(conf.LATEST_TS_PATH):
        with open(conf.LATEST_TS_PATH, encoding='utf-8') as fh:
            ts = fh.read().strip()
    response = make_response(send_file(conf.LATEST_FRAME_PATH, mimetype='image/jpeg'))
    if ts:
        response.headers['X-Data-Timestamp'] = ts
    return response


@app.get('/frames')
def list_frame_sets():
    """Return all timestamps that have saved frame windows."""
    pattern = os.path.join(conf.FRAMES_DIR, 'frame*-*.png')
    files = glob.glob(pattern)
    timestamps = sorted({os.path.basename(p).split('-', 1)[1].rsplit('.', 1)[0] for p in files})
    return jsonify({'timestamps': timestamps})


@app.get('/frames/<timestamp>')
def list_frames(timestamp: str):
    """List frame indices available for a given capture timestamp."""
    pattern = os.path.join(conf.FRAMES_DIR, f'frame*-{timestamp}.png')
    files = sorted(glob.glob(pattern))
    frames = []
    for p in files:
        base = os.path.basename(p)
        idx = int(base.split('-', 1)[0].replace('frame', ''))
        frames.append({'index': idx})
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


def main():
    """Start the metrics thread and run the Flask development server."""
    stop_event = threading.Event()
    thread = start_metrics_thread(stop_event)
    try:
        app.run(host=conf.FLASK_HOST, port=conf.FLASK_PORT)
    finally:
        stop_event.set()
        thread.join()


if __name__ == '__main__':
    main()
