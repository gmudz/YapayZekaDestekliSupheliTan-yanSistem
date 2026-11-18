"""6698 Sayılı KVKK Veri Minimizasyonu Uç-Bulut Gizlilik Köprüsü (TÜBİTAK 2209-A).

Mimari düzeyde mahremiyet odaklı tasarım (Privacy by Design) ilkesini uygular:
- Ham CCTV video akışları yalnızca Uç Katmanında işlenir ve yerel ağ dışına çıkmaz.
- Yalnızca Zamansal ve Yörüngesel şüpheli davranış kriterlerini karşılayan kişilerin
  kırpılmış yüz/vücut görüntüsü ile olay meta verileri Bulut Katmanına aktarılır.
"""

import base64
import time
import cv2
import numpy as np
import requests
from typing import Dict, Optional, Any


class KVKKPrivacyBridge:
    """KVKK veri minimizasyonunu zorunlu kılan Uç-Bulut aktarım köprüsü."""

    def __init__(
        self,
        cloud_api_url: str = "http://127.0.0.1:8000",
        camera_id: str = "KAM-01-GİRİŞ",
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
        """Ham video karesinin yerelde tutulduğunu KVKK denetim sayacına işler."""
        self.blocked_raw_frames_count += 1

    @staticmethod
    def crop_suspect_region(
        frame: np.ndarray, bbox: list, pad_ratio: float = 0.08
    ) -> np.ndarray:
        """Şüpheli kişi/yüz bölgesini güvenli kenar payı ile kırpar."""
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
        """BGR görüntüyü base64 JPEG metnine dönüştürür."""
        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def dispatch_suspicious_event(
        self, frame: np.ndarray, alert: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Şüpheli görüntüyü kırpar, KVKK uyumlu meta verileri ekler ve Bulut Katmanına iletir."""
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
            "pose_action": str(alert.get("pose_action", "BEKLEME")),
            "edge_risk_percent": float(alert.get("risk_percent", 0.0)),
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

        if self.local_cloud_fallback is not None:
            return self.local_cloud_fallback.process_edge_event(payload, crop)

        return {
            "status": "yerel_kuyrukta",
            "kvkk_compliant": True,
            "track_id": alert["track_id"],
            "roi_name": alert["roi_name"],
        }
