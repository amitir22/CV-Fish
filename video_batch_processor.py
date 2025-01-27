#!/usr/bin/env python3
"""
VideoBatchProcessor class: each call to next_batch() reads one new frame,
adds it to a queue, and if the queue exceeds batch_size, it drops the oldest frame.
"""

import cv2
from collections import deque

class VideoBatchProcessor:
    def __init__(self, video_path, batch_size):
        """
        :param video_path: Path to the input video file (e.g. MP4).
        :param batch_size: Maximum number of frames to keep in the sliding window queue.
        """
        self.video_path = video_path
        self.batch_size = batch_size

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        self.frame_queue = deque()
        self.finished = False

    def next_batch(self):
        """
        Reads exactly one new frame from the video, appends it to the sliding window queue.
        If the queue exceeds the batch_size, drops the oldest frame from the front.

        :return: A list of frames (up to batch_size in length), or None if no more frames.
        """
        if self.finished:
            return None  # We've already reached the end in a previous call

        ret, frame = self.cap.read()
        if not ret:
            # No more frames or error reading
            self.finished = True
            return None

        # Add the new frame to the queue
        self.frame_queue.append(frame)

        # If we exceed batch_size, drop the oldest frame
        if len(self.frame_queue) > self.batch_size:
            self.frame_queue.popleft()

        # Return a copy of the current sliding window (it could have up to batch_size frames)
        return list(self.frame_queue)

    def close(self):
        """
        Release the video capture resource.
        """
        self.cap.release()
