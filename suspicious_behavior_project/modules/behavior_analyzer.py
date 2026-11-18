"""Şüpheli Davranış ve Amaçsız Bekleme (Oyalanma) Analiz Modülü (Uç Katmanı — TÜBİTAK 2209-A).

İki temel kriterin aynı anda karşılanmasıyla şüpheli davranış tespiti yapar:
1. Zamansal Kriter: Kişinin tanımlı İlgi Alanı (ROI) içinde belirlenen eşik süreden
   (üretimde 180 sn, canlı demo modunda 5 sn) uzun süre kalması.
2. Yörüngesel Kriter: Hareket Verimliliği = (Net Yer Değiştirme / Toplam Yol Mesafesi)
   değerinin belirlenen eşik değerin (0.30) altında kalması.
"""

import time
from collections import defaultdict
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any

from utils.helpers import ascii_tr


class BehaviorAnalyzer:
    """Kişi yörüngelerini ve bölge içi kalış sürelerini analiz ederek amaçsız beklemeyi tespit eder."""

    def __init__(self, config: dict, demo_mode: bool = False):
        self.config = config
        self.rois = config.get("regions_of_interest", [])
        if demo_mode and "demo_time_threshold_seconds" in config:
            self.time_threshold = float(config["demo_time_threshold_seconds"])
        else:
            self.time_threshold = float(config.get("time_threshold_seconds", 180.0))

        self.efficiency_threshold = float(
            config.get("trajectory_efficiency_threshold", 0.30)
        )
        self.min_path_points = int(config.get("min_path_points", 10))
        alert_cfg = config.get("alert", {})
        self.cooldown_seconds = float(alert_cfg.get("cooldown", 30.0))

        self.track_history: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.time_in_roi: Dict[int, Dict[str, Any]] = defaultdict(
            lambda: {
                "start_time": None,
                "roi_name": None,
                "elapsed": 0.0,
                "efficiency": 1.0,
                "net_displacement": 0.0,
                "total_distance": 0.0,
                "status": "NORMAL",
            }
        )
        self.alerts: Dict[int, Dict[str, Any]] = {}
        self.last_cloud_dispatch: Dict[int, float] = {}

    def is_inside_roi(self, point: Tuple[int, int], roi_name: str) -> bool:
        """Verilen (x, y) noktasının belirtilen çokgen ilgi alanı içinde olup olmadığını kontrol eder."""
        for roi in self.rois:
            if roi["name"] == roi_name:
                pts = np.array(roi["points"], np.int32)
                return (
                    cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False)
                    >= 0
                )
        return False

    @staticmethod
    def compute_trajectory_metrics(
        path: List[Tuple[int, int]]
    ) -> Tuple[float, float, float]:
        """Yörünge için (Verimlilik, Net Yer Değiştirme, Toplam Yol Mesafesi) değerlerini hesaplar."""
        if len(path) < 2:
            return 1.0, 0.0, 0.0

        pts = np.array(path, dtype=np.float64)
        start_point = pts[0]
        end_point = pts[-1]

        net_displacement = float(np.linalg.norm(end_point - start_point))
        diffs = pts[1:] - pts[:-1]
        segment_lengths = np.linalg.norm(diffs, axis=1)
        total_distance = float(np.sum(segment_lengths))

        if total_distance < 15.0:
            efficiency = min(0.05, net_displacement / 100.0)
        else:
            raw_efficiency = net_displacement / (total_distance + 1e-6)
            if net_displacement < 35.0 and len(path) >= 10:
                efficiency = min(raw_efficiency, net_displacement / 150.0)
            else:
                efficiency = raw_efficiency

        return float(np.clip(efficiency, 0.0, 1.0)), net_displacement, total_distance

    def update(
        self,
        tracks: List[Dict[str, Any]],
        current_timestamp: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Takip edilen nesnelerle davranış analizini günceller ve şüpheli uyarıları döndürür."""
        current_alerts: List[Dict[str, Any]] = []
        now = current_timestamp if current_timestamp is not None else time.time()

        for track in tracks:
            track_id = track.get("track_id", track.get("id", 0))
            bbox = track["bbox"]
            centroid = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

            self.track_history[track_id].append(centroid)
            if len(self.track_history[track_id]) > 600:
                self.track_history[track_id] = self.track_history[track_id][-600:]

            matched_roi = None
            for roi in self.rois:
                if self.is_inside_roi(centroid, roi["name"]):
                    matched_roi = roi["name"]
                    break

            state = self.time_in_roi[track_id]
            if matched_roi is not None:
                if state["start_time"] is None or state["roi_name"] != matched_roi:
                    state["start_time"] = now
                    state["roi_name"] = matched_roi
                    self.track_history[track_id] = [centroid]

                elapsed = max(0.0, now - state["start_time"])
                path = self.track_history[track_id]
                efficiency, net_disp, total_dist = self.compute_trajectory_metrics(path)

                state["elapsed"] = elapsed
                state["efficiency"] = efficiency
                state["net_displacement"] = net_disp
                state["total_distance"] = total_dist

                if (
                    elapsed >= self.time_threshold
                    and len(path) >= self.min_path_points
                    and efficiency < self.efficiency_threshold
                ):
                    state["status"] = "ŞÜPHELİ OYALANMA"
                    alert_msg = (
                        f"UYARI: Kişi ID {track_id} - {matched_roi} bölgesinde amaçsız bekleme "
                        f"({elapsed:.1f}sn, Verim={efficiency:.2f})"
                    )

                    last_sent = self.last_cloud_dispatch.get(track_id, 0.0)
                    should_send_cloud = (now - last_sent) >= self.cooldown_seconds
                    if should_send_cloud:
                        self.last_cloud_dispatch[track_id] = now

                    alert_obj = {
                        "message": alert_msg,
                        "track_id": track_id,
                        "roi_name": matched_roi,
                        "dwell_time": round(elapsed, 2),
                        "efficiency": round(efficiency, 3),
                        "net_displacement": round(net_disp, 1),
                        "total_distance": round(total_dist, 1),
                        "bbox": list(bbox),
                        "centroid": centroid,
                        "timestamp": now,
                        "should_send_to_cloud": should_send_cloud,
                    }
                    self.alerts[track_id] = alert_obj
                    current_alerts.append(alert_obj)
                else:
                    state["status"] = "BÖLGEDE"
            else:
                state["start_time"] = None
                state["roi_name"] = None
                state["elapsed"] = 0.0
                state["status"] = "NORMAL"
                self.alerts.pop(track_id, None)

            track["roi_name"] = state["roi_name"]
            track["dwell_time"] = round(state["elapsed"], 1)
            track["efficiency"] = round(state["efficiency"], 2)
            track["behavior_status"] = state["status"]

        return current_alerts

    def draw_roi(self, frame: np.ndarray) -> None:
        """Tanımlı çokgen ilgi alanlarını (ROI) ve aktif durumlarını kare üzerine çizer."""
        overlay = frame.copy()
        alert_rois = {a["roi_name"] for a in self.alerts.values()}

        for roi in self.rois:
            name = roi["name"]
            pts = np.array(roi["points"], np.int32).reshape((-1, 1, 2))
            is_alerting = name in alert_rois

            fill_color = (0, 0, 180) if is_alerting else (180, 90, 0)
            border_color = (0, 0, 255) if is_alerting else (255, 140, 0)

            cv2.fillPoly(overlay, [pts], fill_color)
            cv2.polylines(frame, [pts], isClosed=True, color=border_color, thickness=2)

            status_tag = "ALARM!" if is_alerting else f"Esik: {self.time_threshold:.0f}s"
            label = ascii_tr(f"{name} [{status_tag}]")
            lx, ly = int(pts[0][0][0]), max(22, int(pts[0][0][1]) - 10)
            cv2.putText(
                frame,
                label,
                (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                border_color,
                2,
            )

        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
