"""KVKK Privacy-by-Design Edge-to-Cloud Data Minimization Bridge (TÜBİTAK 2209-A).

Implements Law No. 6698 (KVKK) "Veri Minimizasyonu" (Data Minimization) principle at
the architectural level:
- Raw CCTV video streams are processed exclusively on the Edge Layer and never transmitted.
- Only when a person satisfies both Temporal (ROI dwell time) and Trajectory (movement efficiency)
  loitering criteria is their cropped face/person image + event metadata sent to the Cloud Layer
  for 512-D ArcFace extraction and Milvus Watchlist Cosine Similarity matching.
"""

import base64
import time
import cv2
import numpy as np
import requests
from typing import Dict, Optional, Any


class KVKKPrivacyBridge:
    """Edge-to-Cloud bridge enforcing KVKK data minimization."""

    def __init__(
        self,
        cloud_api_url: str = "http://127.0.0.1:8000",
        camera_id: str = "CAM-01-GIRIS",
        enabled: bool = True,
        local_cloud_fallback: Optional[Any] = None,
    ):
        self.cloud_api_url = cloud_api_url.rstrip("/")
        self.camera_id = camera_id
        self.enabled = enabled
        self.local_cloud_fallback = local_cloud_fallback
        self.transmitted_events_count = 0
        self.blocked_raw_frames_count = 0

    def register_frame_processed_on_edge(self) -> None:
        """Increment counter of raw frames kept strictly on the Edge node (KVKK audit metric)."""
        self.blocked_raw_frames_count += 1

    @staticmethod
    def crop_suspect_region(
        frame: np.ndarray, bbox: list, pad_ratio: float = 0.08
    ) -> np.ndarray:
        """Crop the suspect person/face region from the frame with safe boundary padding."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        px = int(bw * pad_ratio)
        py = int(bh * pad_ratio)

        cx1 = max(0, x1 - px)
        cy1 = max(0, y1 - py)
        cx2 = min(w, x2 + px)
        cy2 = min(h, y2 + py)

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return np.zeros((112, 112, 3), dtype=np.uint8)
        return crop.copy()

    @staticmethod
    def encode_image_b64(image: np.ndarray) -> str:
        """Encode BGR image to base64 JPEG string."""
        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def dispatch_suspicious_event(
        self, frame: np.ndarray, alert: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Crop suspect image, attach KVKK-minimized metadata, and send to Cloud Layer."""
        crop = self.crop_suspect_region(frame, alert["bbox"])
        crop_b64 = self.encode_image_b64(crop)

        payload = {
            "camera_id": self.camera_id,
            "track_id": int(alert["track_id"]),
            "roi_name": str(alert["roi_name"]),
            "dwell_time": float(alert["dwell_time"]),
            "efficiency": float(alert["efficiency"]),
            "timestamp": float(alert.get("timestamp", time.time())),
            "kvkk_data_minimized": True,
            "raw_stream_transmitted": False,
            "cropped_image_b64": crop_b64,
        }

        self.transmitted_events_count += 1

        if self.enabled:
            try:
                resp = requests.post(
                    f"{self.cloud_api_url}/api/v1/events/analyze",
                    json=payload,
                    timeout=2.5,
                )
                if resp.status_code == 200:
                    return resp.json()
            except Exception:
                pass

        # In-process fallback if Cloud FastAPI server is running in-memory or unreachable
        if self.local_cloud_fallback is not None:
            return self.local_cloud_fallback.process_edge_event(payload, crop)

        return {
            "status": "queued_locally",
            "kvkk_compliant": True,
            "track_id": alert["track_id"],
            "roi_name": alert["roi_name"],
        }
