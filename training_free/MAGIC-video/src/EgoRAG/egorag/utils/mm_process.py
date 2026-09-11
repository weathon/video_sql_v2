from typing import Dict, Tuple

import numpy as np


def merge_videos(
    processed_videos: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    """
    Merge all NumPy arrays from processed_videos into a single NumPy array and track frame ranges for each video.

    Args:
        processed_videos (dict): A dictionary where keys are video set names and values are NumPy arrays of frames.

    Returns:
        Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
            - A single NumPy array containing all frames from all videos.
            - A dictionary mapping each video set to its start and end frame indices in the merged array.
    """
    merged_frames = []
    frame_ranges = {}
    current_index = 0

    for video_set, frames in processed_videos.items():
        num_frames = frames.shape[0]  # Number of frames in this video set
        start_index = current_index
        end_index = current_index + num_frames - 1

        # Append frames to the merged list
        merged_frames.append(frames)

        # Record frame range for this video set
        frame_ranges[video_set] = (start_index, end_index)

        # Update the current index
        current_index += num_frames

    # Concatenate all frames into a single NumPy array
    merged_array = (
        np.concatenate(merged_frames, axis=0) if merged_frames else np.array([])
    )

    return merged_array, frame_ranges
