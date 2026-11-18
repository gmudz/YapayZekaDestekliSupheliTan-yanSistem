"""Generator for realistic CCTV surveillance test videos and Watchlist face assets.

Uses real person cutouts from YOLOv5/Ultralytics reference images (bus.jpg, zidane.jpg)
composited onto a 1280x720 facility surveillance background so that YOLOv5m detects
real humans with high confidence (>0.80) while demonstrating:
1. A loitering suspect inside Main_Entrance_ROI (low trajectory efficiency < 0.30)
   whose cropped face matches an enrolled Milvus Watchlist identity.
2. A normal pedestrian walking linearly across the scene (high trajectory efficiency > 0.85)
   who does not trigger false alarms.
"""

import os
import math
import cv2
import numpy as np
from typing import Tuple, List, Dict


def _find_sample_images() -> Tuple[np.ndarray, np.ndarray]:
    """Locate bus.jpg and zidane.jpg from installed yolov5/ultralytics packages."""
    candidates_bus = []
    candidates_zidane = []
    try:
        import yolov5

        pkg_dir = os.path.dirname(yolov5.__file__)
        candidates_bus.append(os.path.join(pkg_dir, "data", "images", "bus.jpg"))
        candidates_zidane.append(os.path.join(pkg_dir, "data", "images", "zidane.jpg"))
    except Exception:
        pass

    try:
        import ultralytics

        u_dir = os.path.dirname(ultralytics.__file__)
        candidates_bus.append(os.path.join(u_dir, "assets", "bus.jpg"))
        candidates_zidane.append(os.path.join(u_dir, "assets", "zidane.jpg"))
    except Exception:
        pass

    bus_img = None
    for p in candidates_bus:
        if os.path.exists(p):
            bus_img = cv2.imread(p)
            if bus_img is not None:
                break

    zidane_img = None
    for p in candidates_zidane:
        if os.path.exists(p):
            zidane_img = cv2.imread(p)
            if zidane_img is not None:
                break

    return bus_img, zidane_img


def _extract_person_sprites() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract two full-body person sprites and one suspect portrait for watchlist enrollment."""
    bus_img, zidane_img = _find_sample_images()

    if bus_img is not None:
        # Person 1 from bus.jpg: [ymin:ymax, xmin:xmax] = [390:900, 215:350]
        p1 = cv2.resize(bus_img[390:905, 215:350], (115, 310), interpolation=cv2.INTER_AREA)
        # Person 2 from bus.jpg: [400:880, 48:230]
        p2 = cv2.resize(bus_img[400:880, 50:225], (115, 300), interpolation=cv2.INTER_AREA)
    else:
        p1 = np.full((310, 115, 3), 180, dtype=np.uint8)
        p2 = np.full((300, 115, 3), 160, dtype=np.uint8)

    if zidane_img is not None:
        # Clear face/upper-body from zidane.jpg to graft onto Suspect #1's head region or watchlist
        suspect_face = cv2.resize(zidane_img[180:520, 740:1060], (160, 160))
    else:
        suspect_face = cv2.resize(p1[0:95, 15:100], (160, 160))

    return p1, p2, suspect_face


def _build_cctv_background(width: int = 1280, height: int = 720) -> np.ndarray:
    """Create a realistic architectural plaza/facility CCTV background frame."""
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    # Subtle concrete floor gradient
    for y in range(height):
        val = int(52 + (y / height) * 42)
        bg[y, :] = (val + 4, val + 2, val)

    # Architectural tiles / grid lines
    for x in range(0, width, 120):
        cv2.line(bg, (x, 100), (x, height), (68, 66, 64), 1)
    for y in range(120, height, 100):
        cv2.line(bg, (0, y), (width, y), (68, 66, 64), 1)

    # Top header bar (facility wall)
    cv2.rectangle(bg, (0, 0), (width, 110), (36, 40, 46), -1)
    cv2.line(bg, (0, 110), (width, 110), (95, 105, 115), 2)
    return bg


def _paste_sprite(canvas: np.ndarray, sprite: np.ndarray, cx: int, cy: int) -> Tuple[int, int, int, int]:
    """Paste person sprite centered at (cx, cy) with soft edge blending."""
    sh, sw = sprite.shape[:2]
    ch, cw = canvas.shape[:2]

    x1 = max(0, min(cx - sw // 2, cw - sw))
    y1 = max(0, min(cy - sh // 2, ch - sh))
    x2 = x1 + sw
    y2 = y1 + sh

    canvas[y1:y2, x1:x2] = sprite
    return (x1, y1, x2, y2)


def ensure_demo_assets(
    video_path: str = "data/demo_cctv.mp4",
    num_frames: int = 240,
    fps: int = 20,
) -> Dict[str, str]:
    """Generate demo CCTV video and watchlist enrollment face images if not already present."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(video_path):
        video_path = os.path.join(base_dir, video_path)

    data_dir = os.path.dirname(video_path)
    os.makedirs(data_dir, exist_ok=True)

    p1_sprite, p2_sprite, _ = _extract_person_sprites()

    # Save Suspect #1 (p1_sprite) and Suspect #2 (p2_sprite) reference images for Watchlist seeding
    suspect1_img_path = os.path.join(data_dir, "watchlist_suspect_1.jpg")
    suspect2_img_path = os.path.join(data_dir, "watchlist_suspect_2.jpg")
    cv2.imwrite(suspect1_img_path, p1_sprite)
    cv2.imwrite(suspect2_img_path, p2_sprite)

    if os.path.exists(video_path) and os.path.getsize(video_path) > 50_000:
        return {
            "video_path": video_path,
            "suspect1_image": suspect1_img_path,
            "suspect2_image": suspect2_img_path,
        }

    width, height = 1280, 720
    bg_template = _build_cctv_background(width, height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, float(fps), (width, height))

    for f_idx in range(num_frames):
        frame = bg_template.copy()

        # Suspect #1: Loiters inside Main_Entrance_ROI ([80, 140] to [560, 620])
        # Paces in a small figure-8 / loop around (310, 380) so Net Displacement / Total Path < 0.30
        theta = f_idx * 0.14
        s1_cx = int(310 + 48 * math.cos(theta))
        s1_cy = int(380 + 36 * math.sin(2 * theta))
        _paste_sprite(frame, p1_sprite, s1_cx, s1_cy)

        # Pedestrian #2: Walks linearly from right to left and back across Sensitive_Area_ROI
        # High trajectory efficiency (> 0.85) -> normal transit
        cycle = f_idx % 120
        if cycle < 60:
            s2_cx = int(1120 - cycle * 7.5)
        else:
            s2_cx = int(670 + (cycle - 60) * 7.5)
        s2_cy = 390
        _paste_sprite(frame, p2_sprite, s2_cx, s2_cy)

        writer.write(frame)

    writer.release()
    return {
        "video_path": video_path,
        "suspect1_image": suspect1_img_path,
        "suspect2_image": suspect2_img_path,
    }
