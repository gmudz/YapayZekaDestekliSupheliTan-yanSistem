"""Utility functions for video processing, configuration validation, and visualization."""

import os
import cv2
import yaml
import numpy as np
from typing import Dict, List, Tuple, Any


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    if not os.path.exists(config_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        alt_path = os.path.join(base_dir, config_path)
        if os.path.exists(alt_path):
            config_path = alt_path
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration dictionary keys."""
    required_keys = ["camera", "model", "tracking", "loitering_analysis"]
    for key in required_keys:
        if key not in config and (key != "model" or "models" not in config):
            print(f"Error: Missing required configuration key: {key}")
            return False
    return True


def draw_detection(
    frame: np.ndarray,
    bbox: List[int],
    label: str,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> None:
    """Draw a detection bounding box and label on the frame."""
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
    )


def draw_alert(
    frame: np.ndarray,
    alert_msg: str,
    position: Tuple[int, int] = (20, 45),
    color: Tuple[int, int, int] = (0, 0, 255),
) -> None:
    """Draw a high-visibility alert banner on the frame."""
    x, y = position
    (tw, th), _ = cv2.getTextSize(alert_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    cv2.rectangle(
        frame,
        (max(0, x - 8), max(0, y - th - 10)),
        (min(frame.shape[1], x + tw + 12), min(frame.shape[0], y + 10)),
        (15, 15, 60),
        -1,
    )
    cv2.rectangle(
        frame,
        (max(0, x - 8), max(0, y - th - 10)),
        (min(frame.shape[1], x + tw + 12), min(frame.shape[0], y + 10)),
        color,
        2,
    )
    cv2.putText(
        frame, alert_msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2
    )


def load_camera_source(config: Dict[str, Any]) -> cv2.VideoCapture:
    """Load and configure video capture source with automatic fallback to demo CCTV video."""
    source = config["camera"]["source"]
    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return cap
        cap.release()

    fallback = config["camera"].get("fallback_video", "data/demo_cctv.mp4")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(fallback):
        fallback = os.path.join(base_dir, fallback)

    if not os.path.exists(fallback):
        from utils.demo_video_generator import ensure_demo_assets

        ensure_demo_assets(fallback)

    cap = cv2.VideoCapture(fallback)
    if not cap.isOpened():
        raise ValueError(f"Failed to open camera source {source} and fallback {fallback}")
    return cap


def get_frame_size(cap: cv2.VideoCapture) -> Tuple[int, int]:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return width, height


def normalize_roi_points(
    roi_points: List[List[int]], frame_size: Tuple[int, int]
) -> List[List[int]]:
    width, height = frame_size
    normalized_points = []
    for point in roi_points:
        x = max(0, min(int(point[0]), width - 1))
        y = max(0, min(int(point[1]), height - 1))
        normalized_points.append([x, y])
    return normalized_points


def preprocess_frame(frame: np.ndarray, size: Tuple[int, int] = (1280, 720)) -> np.ndarray:
    if frame is None:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    if (frame.shape[1], frame.shape[0]) != size:
        return cv2.resize(frame, size)
    return frame


def draw_boxes(
    frame: np.ndarray, boxes: List[List[int]], labels: List[str]
) -> np.ndarray:
    for box, label in zip(boxes, labels):
        draw_detection(frame, box, label)
    return frame
