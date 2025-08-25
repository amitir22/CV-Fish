"""Central configuration for the CV-Fish project.

This module centralises all tunable parameters used throughout the
application.  It includes default keyword arguments for the optical flow
metric extractors as well as runtime values that control sampling
behaviour and the live Bollinger GUI.

Each constant is defined using :class:`typing.Final` to communicate that
the value is intended to be immutable.  If you need to modify a
configuration at runtime, make a copy instead of mutating these
structures directly.
"""

import os
import cv2
from frozendict import frozendict
from typing import Final, Tuple

# =============================================================================
# Metrics extraction:
# =============================================================================
# NOTE: if you want to edit the params, edit their copy with:
#       >>> editable_params = dict(DEFAULT_<*>_PARAMS)
FAST_NL_MEANS_DENOISING_KWARGS = frozendict({
    "h": 10,
    "templateWindowSize": 7,
    "searchWindowSize": 21
})

DEFAULT_FARNEBACK_PARAMS = frozendict({
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 15,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
    "should_apply_gaussian_denoiser": False
})

DEFAULT_TVL1_PARAMS = frozendict({"should_apply_gaussian_denoiser": False})

DEFAULT_LUCAS_KANADE_PARAMS = frozendict({
    "win_size": (15, 15),
    "max_level": 2,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    "max_corners": 100,
    "quality_level": 0.3,
    "min_distance": 7,
    "block_size": 7,
    "should_apply_gaussian_denoiser": False
})


# =============================================================================
# Live Bollinger GUI:
# =============================================================================
QUIVER_KWARGS = frozendict({
    "color": 'red',
    "units": 'xy',
    "angles": 'xy',
    "scale_units": 'xy',

    # Make arrows bigger (smaller 'scale' => longer arrows)
    "scale": 0.05,
    
    # Pivot: 'tail' => (X,Y) is at arrow tail; 'mid' => arrow centered on (X,Y);
    #        'tip' => arrow tip is at (X,Y).
    "pivot": 'tail',

    # Increase arrow thickess
    "width": 0.005,

    # Increase arrowhead size
    "headwidth": 10,
    "headlength": 12,
    "headaxislength": 8,

    # Optional: outline thickness and color
    "linewidths": 0.7,
    "units": 'xy',

    "alpha": 1.0,  # fully opaque
    "zorder": 2    # drawn above the image
})


# NOTE: Final[type] == const.
# =============================================================================
# Main Parameters:
# =============================================================================
T_WINDOW: Final[int] = 100
"""Length of the rolling time window (seconds) displayed in the Bollinger chart."""

BOLLINGER_NUM_STD_OF_BANDS: Final[float] = 2.0
"""Number of standard deviations used to draw Bollinger bands."""

DEFAULT_SUPER_PIXEL_SHAPE: Final[Tuple[int, int]] = (1, 1)
"""Factor by which incoming frames are downscaled in height and width."""

WEBCAM_RETRY_INTERVAL_SECONDS: Final[int] = 2
"""The frame polling interval when failing to retrieve a frame from the capture device."""

# Number of frames to capture in each sampling window
FRAME_WINDOW_SIZE: Final[int] = 5

# Interval between capture sessions in minutes
CAPTURE_INTERVAL_MINUTES: Final[int] = 10

DEFAULT_SUPER_PIXEL_DIMEMSNIONS = (4, 4)
"""Default dimensions used when downscaling frames for processing."""


# =============================================================================
# I/O paths and outputs:
# =============================================================================
OUTPUT_DIR: Final[str] = "./output"
# Latest frame saved by the metrics worker; stored as PNG to align with
# persistent frame-window naming (frame<idx>-YYYYMMDD-hhmm.png)
LATEST_FRAME_PATH: Final[str] = os.path.join(OUTPUT_DIR, "latest_frame.png")
LATEST_TS_PATH: Final[str] = os.path.join(OUTPUT_DIR, "latest_frame_timestamp.txt")
FRAMES_DIR: Final[str] = os.path.join(OUTPUT_DIR, "frames")


# =============================================================================
# Video source configuration:
# =============================================================================
VIDOE_FILE_PATH: Final[str] = './Workable Data/Processed/DPH21_Above_IR10.avi'
NVR_USER: Final[str] = 'admin'
NVR_PASS: Final[str] = 'admin12345'
NVR_IP: Final[str] = '0.0.0.0'  # within network
NVR_PORT: Final[str] = '554'  # default port for the protocol, might not need change
NVR_PATH: Final[str] = '/Streaming/Channels/101'
VIDEO_SOURCE = frozendict({
    'FILE': VIDOE_FILE_PATH,
    'WEBCAM': 0,
    'NVR': f"rtsp://{NVR_USER}:{NVR_PASS}@{NVR_IP}:{NVR_PORT}{NVR_PATH}"
})


# =============================================================================
# Flask API configuration:
# =============================================================================
FLASK_HOST: Final[str] = '0.0.0.0'
FLASK_PORT: Final[int] = 5000
