import os

from oslactionspotting.core.utils.io import load_text


def load_classes(input):
    """Load classes from either list or txt file.

    Args: 
        input (string): Path of the file that contains one class per line or list of classes.

    Returns:
        Dictionnary with classes associated to indexes.
    """
    if isinstance(input, list):
        return {x: i + 1 for i, x in enumerate(sorted(input))}
    return {x: i + 1 for i, x in enumerate(load_text(input))}


# def read_fps(video_frame_dir):
#     with open(os.path.join(video_frame_dir, 'fps.txt')) as fp:
#         return float(fp.read())


# def get_num_frames(video_frame_dir):
#     max_frame = -1
#     for img_file in os.listdir(video_frame_dir):
#         if img_file.endswith('.jpg'):
#             frame = int(os.path.splitext(img_file)[0])
#             max_frame = max(frame, max_frame)
#     return max_frame + 1
