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
BOLLINGER_NUM_STD_OF_BANDS: Final[float] = 2.0
DEFAULT_SUPER_PIXEL_SHAPE: Final[Tuple[int, int]] = (1, 1)
WEBCAM_RETRY_INTERVAL_SECONDS: Final[int] = 2  # The frame polling interval when failing to retrieve a frame
FRAME_WINDOW_SIZE: Final[int] = 5  # Number of frames captured each interval
CAPTURE_INTERVAL_MINUTES: Final[int] = 10  # Interval between capture sessions
DEFAULT_SUPER_PIXEL_DIMEMSNIONS = (4, 4)
