import os
import cv2
import json

from oslactionspotting.core.utils.io import load_json


def check_config(cfg):
    # assert cfg.dataset.test.path is not None and os.path.isfilecfg.dataset.test.path.
    return


class Visualizer:
    """Visualier class used to visualize a video along with its predictions using opencv.

    Args:
        cfg: Config dict. It should at least contain the key dataset.test.results which contains the predictions file of the video/feature infered.
        The key dataset.test.path which is the video path and the key visualize which is a dict for usefulr informations to visualize the video.
    """

    def __init__(self, cfg):
        check_config(cfg)
        self.video_path = cfg.dataset.test.path
        assert cfg.dataset.test.results.endswith(".json")

        data = load_json(os.path.join(cfg.work_dir, cfg.dataset.test.results))

        predictions = data["predictions"]

        # Keep only predictions above the chosen treshold
        self.prediction_list = [
            (pred["position"], pred["label"], pred["confidence"])
            for pred in predictions
            if pred["confidence"] > cfg.visualizer.threshold
        ]

        self.cap = cv2.VideoCapture(self.video_path)
        assert self.cap.isOpened()

        # Range for the annotation in the video, the action will be displayed within the range (in ms).
        self.annotation_range = cfg.visualizer.annotation_range

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        # Seconds of the video that are skipped when typing on the touch p.
        seconds_to_skip = cfg.visualizer.seconds_to_skip
        self.frames_to_skip = int(self.fps * seconds_to_skip)
        # Scaling of the dimensions of the frames of the video.
        self.scale = cfg.visualizer.scale

    def visualize(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Get the current position in milliseconds
            current_time = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            frame = cv2.resize(
                frame,
                None,
                fx=self.scale,
                fy=self.scale,
                interpolation=cv2.INTER_LINEAR,
            )

            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]

            # Collect all annotations within the time range
            annotations = []
            for position, label, confidence in self.prediction_list:
                if abs(current_time - position) <= self.annotation_range:
                    annotations.append((label, confidence))

            # Sort all annotations collected for the time range by their confidences
            annotations = sorted(annotations, key=lambda x: x[1], reverse=True)

            if annotations:
                # Display only the first three annotations with the highest score.
                for i, (label, confidence) in enumerate(annotations[:3]):

                    text = f"{label} ({confidence:.2f})"

                    text_size = cv2.getTextSize(text, 1, 1, 2)[0]

                    x_coordinate = (frame_width - text_size[0]) // 2

                    y_coordinate = frame_height - 15 - (i * (text_size[1] + 10))

                    position_text = (x_coordinate, y_coordinate)

                    cv2.putText(frame, text, position_text, 1, 1, (0, 0, 255), 2)

            # Display the frame
            cv2.imshow("Video", frame)

            # Press Q on keyboard to  exit or p to skip an amount of seconds in the video
            key = cv2.waitKey(int(self.fps)) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                new_frame_position = current_frame + self.frames_to_skip
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_position)
            elif key == ord("a"):
                current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                new_frame_position = current_frame - self.frames_to_skip
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_position)
