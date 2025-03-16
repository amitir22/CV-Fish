#!/usr/bin/env python3

import sys
import cv2 as cv
import numpy as np
import torch
from tqdm import tqdm


def dense_optical_flow_to_tensor(cap_object: cv.VideoCapture, device='cpu'):
    """
    Compute dense optical flow for consecutive frames of a given video file
    and store them in a single PyTorch tensor.

    :param video_path: Path to the video file (e.g., .avi, .mp4, etc.)
    :param device: 'cpu' or 'cuda' (if GPU support is desired for storing the final tensor)
    :return: A torch.Tensor of shape [num_frames-1, 2, height, width]
    """
    # Try to get the total number of frames.
    total_frames = int(cap_object.get(cv.CAP_PROP_FRAME_COUNT))

    # Read the first frame
    ret, frame1 = cap_object.read()
    if not ret:
        cap_object.release()
        raise IOError("Cannot read the first frame from the video.")

    # Convert to grayscale
    prev_gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

    flow_list = []

    # Prepare a progress bar if total_frames is valid; otherwise fallback.
    if total_frames > 0:
        # We'll compute optical flow for consecutive frame pairs,
        # so the number of optical flow fields is total_frames - 1
        pbar = tqdm(total=(total_frames - 1), desc="Computing optical flow")
    else:
        # If frame count is unknown (-1), we can still use tqdm in 'indeterminate' mode.
        pbar = tqdm(desc="Computing optical flow")

    frame_index = 1  # We already grabbed the 1st frame
    while True:
        ret, frame2 = cap_object.read()
        if not ret:
            # No more frames
            break

        next_gray = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

        # Calculate dense optical flow (Farneback)
        flow = cv.calcOpticalFlowFarneback(
            prev_gray, next_gray, None,
            0.5,  # pyr_scale
            3,    # levels
            15,   # winsize
            3,    # iterations
            5,    # poly_n
            1.2,  # poly_sigma
            0     # flags
        )
        # flow shape: (H, W, 2)

        # Convert flow to (2, H, W) for PyTorch convention
        flow_t = np.transpose(flow, (2, 0, 1))

        flow_list.append(flow_t)

        prev_gray = next_gray
        frame_index += 1

        # TODO: DELETE
        if frame_index == 200:
            break

        # Update progress bar
        pbar.update(1)

    pbar.close()
    cap_object.release()

    if not flow_list:
        raise ValueError("No flow data was computed (video may have fewer than 2 frames).")

    # Stack into a single tensor: [num_frames-1, 2, H, W]
    flow_tensor = torch.from_numpy(np.stack(flow_list, axis=0)).float()
    flow_tensor = flow_tensor.to(device)

    return flow_tensor


def main():
    """
    main
        Main function.
        Usage: 
        python dense_flow_to_torch.py <video_file.avi> [device]
    """
    if len(sys.argv) < 2:
        print("Usage: python dense_flow_to_torch.py <video_file.avi> [device]")
        print("Example: python dense_flow_to_torch.py input.avi cuda")
        sys.exit(1)

    video_path = sys.argv[1]
    device = 'cpu' if len(sys.argv) < 3 else sys.argv[2]

    # Initializing a capture object
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    # Compute the flow tensor
    print("Computing output tensor")
    flow_tensor = dense_optical_flow_to_tensor(video_path, device=device)

    print(f"Computed flow for {video_path}.")
    print(f"Flow tensor shape: {flow_tensor.shape} (frames, 2, H, W)")
    print(f"Flow tensor device: {flow_tensor.device}")
    # e.g. shape might be [X, 2, 720, 1280] for a 720p video.

    # Now you can proceed to use 'flow_tensor' in your PyTorch pipeline,
    # or save it, e.g.:
    torch.save(flow_tensor, f"{video_path}.flow_fields.pt")


if __name__ == "__main__":
    main()
