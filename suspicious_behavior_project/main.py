"""Main application for TÜBİTAK 2209-A Hybrid Edge/Cloud Suspicious Behavior & Fair Face Recognition System.

Integrates:
1. Edge Layer:
   - YOLOv5m Real-Time Person Detection (modules/person_detector.py)
   - DeepSORT Tracking with 8-D Kalman Filter & 512-D Re-ID Appearance Matching (modules/object_tracker.py)
   - Dual-Criterion Loitering Analysis: ROI Dwell Time + Trajectory Efficiency (modules/behavior_analyzer.py)
   - KVKK Privacy-by-Design Data Minimization Bridge (modules/privacy_bridge.py)
2. Cloud Layer:
   - FastAPI Cloud Service & Live Web Dashboard on port 8000 (cloud/server.py)
   - 512-D ArcFace Face Embedding with FairFace Demographic Bias Mitigation (modules/face_recognizer.py)
   - Milvus Vector Database Watchlist Search via Cosine Similarity (modules/vector_db.py)
"""

import os
import sys
import time
import base64
import argparse
import threading
import cv2
import yaml
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from modules.person_detector import PersonDetector
from modules.object_tracker import ObjectTracker
from modules.behavior_analyzer import BehaviorAnalyzer
from modules.privacy_bridge import KVKKPrivacyBridge
from modules.fairness_evaluator import TubitakEvaluationSuite
from utils.helpers import load_config, load_camera_source, draw_alert


def convert_detections(
    detections: List[Tuple[int, int, int, int, float]]
) -> Tuple[List[List[float]], List[float]]:
    """Convert detections from (xmin, ymin, xmax, ymax, conf) to ([x, y, w, h], conf)."""
    xywh_detections = []
    confidences = []
    for det in detections:
        xmin, ymin, xmax, ymax = det[:4]
        conf = det[4] if len(det) >= 5 else 0.90
        width = xmax - xmin
        height = ymax - ymin
        xywh_detections.append([xmin, ymin, width, height])
        confidences.append(conf)
    return xywh_detections, confidences


def draw_detections(
    frame: np.ndarray,
    tracked_objects: List[Dict[str, Any]],
    watchlist_matches: Optional[Dict[int, Dict[str, Any]]] = None,
) -> None:
    """Draw tracked persons, trajectory trails, ROI dwell telemetry, and Cloud Watchlist hits."""
    watchlist_matches = watchlist_matches or {}

    for obj in tracked_objects:
        bbox = obj["bbox"]
        track_id = obj["track_id"]
        status = obj.get("behavior_status", "NORMAL")
        dwell = obj.get("dwell_time", 0.0)
        eff = obj.get("efficiency", 1.0)

        if status == "LOITERING":
            color = (0, 0, 255)  # Red for confirmed suspicious loitering
        elif status == "IN_ROI":
            color = (0, 180, 255)  # Amber for monitored inside ROI
        else:
            color = (0, 220, 90)  # Green for normal transit

        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 4, color, -1)

        # Draw trajectory trail
        history = obj.get("history", [])
        if len(history) > 1:
            pts = history[-40:]
            for i in range(1, len(pts)):
                if pts[i] is not None and pts[i - 1] is not None:
                    cv2.line(frame, pts[i - 1], pts[i], color, 2)

        # Primary label: Track ID + Status
        header = f"ID:{track_id} [{status}]"
        cv2.putText(
            frame,
            header,
            (x1, max(20, y1 - 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
        )

        # Secondary telemetry: Dwell time & Trajectory efficiency
        telemetry = f"Sure:{dwell:.1f}s | Verim:{eff:.2f}"
        cv2.putText(
            frame,
            telemetry,
            (x1, max(38, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (235, 245, 255),
            1,
        )

        # If Cloud Layer matched this Track ID in the Milvus Watchlist, display badge
        if track_id in watchlist_matches:
            match_info = watchlist_matches[track_id]
            match_text = (
                f"MILVUS MATCH: {match_info['name']} (Cos:{match_info['cosine_similarity']:.2f})"
            )
            cv2.rectangle(
                frame,
                (x1, min(frame.shape[0] - 28, y2 + 4)),
                (min(frame.shape[1] - 5, x1 + 380), min(frame.shape[0] - 2, y2 + 28)),
                (0, 0, 190),
                -1,
            )
            cv2.putText(
                frame,
                match_text,
                (x1 + 5, min(frame.shape[0] - 8, y2 + 21)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
            )


def start_cloud_server_in_background(host: str = "0.0.0.0", port: int = 8000):
    """Launch the FastAPI Cloud Layer & Web Dashboard in a daemon thread if not already bound."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            print(f"[Cloud Layer] FastAPI Web Dashboard already running at http://127.0.0.1:{port}")
            from cloud.server import cloud_engine

            return cloud_engine

    import uvicorn
    from cloud.server import app, cloud_engine

    def _run():
        uvicorn.run(app, host=host, port=port, log_level="warning")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    time.sleep(1.0)
    print(f"[Cloud Layer] FastAPI & Web Dashboard started at http://127.0.0.1:{port}")
    return cloud_engine


def print_evaluation_report(report: Dict[str, Any]) -> None:
    """Print formatted TÜBİTAK 2209-A Hedef 1-4 benchmark report to console."""
    print("\n" + "=" * 86)
    print(" TÜBİTAK 2209-A UÇTAN UCA (E2E) DOĞRULAMA VE ADİLLİK DENETİMİ RAPORU")
    print("=" * 86)
    h1 = report["hedef_1_performance"]
    h2 = report["hedef_2_behavior"]
    h3 = report["hedef_3_face_recognition"]
    h4 = report["hedef_4_fairness"]

    print(
        f" [HEDEF 1] {h1['target_name']}\n"
        f"   -> Eşzamanlı Akış: {h1['concurrent_streams']}x {h1['resolution']} | "
        f"Akış Başına FPS: {h1['avg_fps_per_stream']} FPS (Toplam: {h1['total_system_fps']} FPS) | "
        f"Durum: {'BAŞARILI ✅' if h1['passed'] else 'BAŞARISIZ ❌'}"
    )
    print(
        f" [HEDEF 2] {h2['target_name']}\n"
        f"   -> Precision: {h2['precision']} | Recall: {h2['recall']} | "
        f"F1-Skoru: {h2['f1_score']} (Hedef >= 0.80) | "
        f"Durum: {'BAŞARILI ✅' if h2['passed'] else 'BAŞARISIZ ❌'}"
    )
    print(
        f" [HEDEF 3] {h3['target_name']}\n"
        f"   -> FAR={h3['far_threshold']} Eşiğinde Temel FRR: %{h3['baseline_frr_percent']} -> "
        f"FairFace FRR: %{h3['fairface_frr_percent']} (EER: %{h3['eer_percent']}) | "
        f"Durum: {'BAŞARILI ✅' if h3['passed'] else 'BAŞARISIZ ❌'}"
    )
    print(
        f" [HEDEF 4] {h4['target_name']}\n"
        f"   -> Gruplar Arası Max FNMR Farkı: %{h4['baseline_max_gap_percent']} -> "
        f"%{h4['fairface_max_gap_percent']} (Azalma: %{h4['disparity_reduction_percent']}) | "
        f"Durum: {'BAŞARILI ✅' if h4['passed'] else 'BAŞARISIZ ❌'}"
    )
    print("=" * 86 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="TÜBİTAK 2209-A Hibrit Şüpheli Davranış Tespiti ve Adil Yüz Tanıma Sistemi"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Yapılandırma dosyası yolu"
    )
    parser.add_argument(
        "--source", type=str, default=None, help="Kamera kaynağı (0, video.mp4, rtsp://...)"
    )
    parser.add_argument(
        "--production-timers",
        action="store_true",
        help="180 saniyelik tam üretim bekleme eşiğini kullan (varsayılan: canlı demo için 5 sn)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="OpenCV penceresi açmadan yalnızca Web Gösterge Paneli ve konsol ile çalıştır",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Belirtilen kare sayısından sonra otomatik çık (0 = sınırsız)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="TÜBİTAK 2209-A Hedef 1-4 doğrulama testlerini çalıştır ve çık",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.source is not None:
        config["camera"]["source"] = int(args.source) if args.source.isdigit() else args.source

    if args.evaluate:
        suite = TubitakEvaluationSuite(config)
        report = suite.run_all_benchmarks()
        print_evaluation_report(report)
        return

    # Start Cloud Layer FastAPI + Web Dashboard on port 8000
    cloud_cfg = config.get("cloud", {})
    cloud_engine = start_cloud_server_in_background(
        host=cloud_cfg.get("host", "0.0.0.0"),
        port=int(cloud_cfg.get("port", 8000)),
    )

    # Print initial TÜBİTAK benchmark verification summary
    if cloud_engine.latest_benchmark:
        print_evaluation_report(cloud_engine.latest_benchmark)

    # Initialize Edge Layer components
    cap = load_camera_source(config)

    detector = PersonDetector(
        model_name=config["model"]["weights"],
        confidence_threshold=config["model"]["confidence_threshold"],
        iou_threshold=config["model"].get("iou_threshold", 0.45),
        device=config["model"].get("device", "auto"),
        half_precision=config["model"].get("half_precision", False),
        input_size=config["model"].get("input_size", 416),
    )

    tracker = ObjectTracker(
        max_disappeared=config["tracking"]["max_age"],
        max_distance=config["tracking"].get("max_distance", 140.0),
        min_hits=config["tracking"].get("min_hits", 2),
        iou_threshold=config["tracking"].get("iou_threshold", 0.25),
        appearance_weight=config["tracking"].get("appearance_weight", 0.45),
        feature_dim=config["tracking"].get("feature_dim", 512),
    )

    demo_mode = not args.production_timers
    behavior_analyzer = BehaviorAnalyzer(
        config["loitering_analysis"], demo_mode=demo_mode
    )

    privacy_bridge = KVKKPrivacyBridge(
        cloud_api_url=cloud_cfg.get("api_url", "http://127.0.0.1:8000"),
        camera_id=config["camera"].get("camera_id", "CAM-01-GIRIS"),
        enabled=True,
        local_cloud_fallback=cloud_engine,
    )

    can_show_gui = (not args.headless) and bool(
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )
    window_name = "TUBITAK 2209-A | Supheli Davranis ve Adil Yuz Tanima Sistemi"
    if can_show_gui:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720)
        except Exception:
            can_show_gui = False

    print(
        f"[Edge Layer] Pipeline started (YOLOv5m on {detector.device_str} + DeepSORT + "
        f"Loitering Threshold={behavior_analyzer.time_threshold}s + KVKK Bridge)"
    )
    print(f"[Web Dashboard] Open http://127.0.0.1:{cloud_cfg.get('port', 8000)} in browser")

    fps = 25.0
    frame_idx = 0
    watchlist_matches: Dict[int, Dict[str, Any]] = {}
    last_bag_cloud_alert: Dict[int, float] = {}
    sim_clock = time.time()
    live_webcam_enrolled = False

    # Asynchronous background detector state for smooth 20-30 FPS live webcam rendering
    det_lock = threading.Lock()
    shared_det_state: Dict[str, Any] = {
        "frame_to_process": None,
        "latest_dets": [],
        "running": True,
    }

    def _async_detector_worker():
        while shared_det_state["running"]:
            frm = shared_det_state["frame_to_process"]
            if frm is not None:
                shared_det_state["frame_to_process"] = None
                res = detector.detect_with_conf(frm)
                with det_lock:
                    shared_det_state["latest_dets"] = res
            else:
                time.sleep(0.01)

    det_thread = threading.Thread(target=_async_detector_worker, daemon=True)
    det_thread.start()

    try:
        while True:
            t_frame_start = time.perf_counter()

            # Check interactive Web Dashboard controls
            ctrl = getattr(cloud_engine, "runtime_controls", {})
            if ctrl.get("requested_source") is not None:
                req_src = ctrl["requested_source"]
                ctrl["requested_source"] = None
                try:
                    cap.release()
                except Exception:
                    pass
                if req_src in ("0", "webcam"):
                    config["camera"]["source"] = 0
                else:
                    config["camera"]["source"] = config["camera"].get(
                        "fallback_video", "data/demo_cctv.mp4"
                    )
                cap = load_camera_source(config)

            if ctrl.get("time_threshold") is not None:
                behavior_analyzer.time_threshold = float(ctrl["time_threshold"])
                ctrl["time_threshold"] = None

            if ctrl.get("efficiency_threshold") is not None:
                behavior_analyzer.efficiency_threshold = float(ctrl["efficiency_threshold"])
                ctrl["efficiency_threshold"] = None

            ret, frame = cap.read()
            if not ret or frame is None:
                # Loop video seamlessly for continuous surveillance demonstration
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Video stream ended.")
                    break

            if frame.shape[1] != 1280 or frame.shape[0] != 720:
                frame = cv2.resize(frame, (1280, 720))

            frame_idx += 1
            privacy_bridge.register_frame_processed_on_edge()

            is_live_webcam = str(config["camera"].get("source", "0")) in ("0", "/dev/video0")

            # Use real wall-clock seconds on live webcam, or 0.20s/frame simulation clock on demo video
            if is_live_webcam or not demo_mode:
                sim_clock = time.time()
            else:
                sim_clock += 0.20

            # 1. Edge Person Detection (YOLOv5m)
            if is_live_webcam and args.max_frames == 0:
                # Submit frame to background worker and read latest detections without blocking webcam FPS
                if shared_det_state["frame_to_process"] is None:
                    shared_det_state["frame_to_process"] = frame.copy()
                with det_lock:
                    dets_with_conf = list(shared_det_state["latest_dets"])
                if frame_idx == 1 and not dets_with_conf:
                    dets_with_conf = detector.detect_with_conf(frame)
                    with det_lock:
                        shared_det_state["latest_dets"] = dets_with_conf
            else:
                dets_with_conf = detector.detect_with_conf(frame)

            det_boxes = [(d[0], d[1], d[2], d[3]) for d in dets_with_conf]
            det_confs = [d[4] for d in dets_with_conf]

            # 2. Edge Multi-Object Tracking (DeepSORT: Kalman + Re-ID Appearance)
            tracked_objects = tracker.update(
                det_boxes, frame=frame, confidences=det_confs
            )

            # Auto-enroll live webcam user into Milvus Watchlist on first clear detection so live match works immediately
            if is_live_webcam and not live_webcam_enrolled and tracked_objects:
                try:
                    trk0 = tracked_objects[0]
                    crop0 = privacy_bridge.crop_suspect_region(frame, trk0["bbox"])
                    aligned0, _, _ = cloud_engine.recognizer.detect_and_align_face(crop0)
                    emb0 = cloud_engine.recognizer.extract_embedding(aligned0, already_aligned=True)
                    b64_0 = privacy_bridge.encode_image_b64(aligned0)
                    cloud_engine.vector_db.enroll_identity(
                        person_id="WL-LIVE-CAM",
                        name="Canlı Kamera Kullanıcısı (Aktif İzleme)",
                        embedding=emb0,
                        risk_level="CRITICAL",
                        demographic_group="Middle_Eastern",
                        image_b64=b64_0,
                        notes="Canlı laptoptan otomatik izleme listesi kaydı",
                    )
                    live_webcam_enrolled = True
                    print("[Live Webcam] Canlı kamera yüzü Milvus İzleme Listesine (WL-LIVE-CAM) kaydedildi!")
                except Exception as e:
                    print(f"[Live Webcam] Auto-enroll warning: {e}")

            # 3. Edge Behavior Analysis (ROI Dwell Time + Trajectory Efficiency)
            alerts = behavior_analyzer.update(
                tracked_objects, current_timestamp=sim_clock
            )

            # Handle instant test trigger button from Web Dashboard
            if ctrl.get("force_trigger_alert") and tracked_objects:
                ctrl["force_trigger_alert"] = False
                target_trk = tracked_objects[0]
                target_trk["behavior_status"] = "LOITERING"
                alerts.append(
                    {
                        "message": f"ALERT: Person ID {target_trk['track_id']} loitering in Main_Entrance_ROI (Manual Test)",
                        "track_id": target_trk["track_id"],
                        "roi_name": target_trk.get("roi_name") or "Main_Entrance_ROI",
                        "dwell_time": max(behavior_analyzer.time_threshold + 2.5, target_trk.get("dwell_time", 6.5)),
                        "efficiency": 0.14,
                        "bbox": target_trk["bbox"],
                        "centroid": target_trk["centroid"],
                        "timestamp": sim_clock,
                        "should_send_to_cloud": True,
                    }
                )

            # Keep an untouched copy of the frame for Cloud Layer suspect face cropping
            clean_frame_for_cloud = frame.copy()

            # 4. Smart AI Extensions (KVKK Dynamic Face Blur, Pose Skeleton, Heatmap, Future Trajectory, Abandoned Objects)
            smart_engine = getattr(cloud_engine, "smart_engine", None)
            if smart_engine is not None:
                smart_engine.apply_dynamic_kvkk_masking(frame, tracked_objects)
                pre_breach_warnings = smart_engine.update_and_draw_heatmap_and_predictions(
                    frame, tracked_objects, behavior_analyzer.rois
                )
                smart_engine.analyze_and_draw_pose_actions(frame, tracked_objects)
                bag_alerts = smart_engine.update_and_draw_abandoned_objects(
                    frame,
                    getattr(detector, "last_bag_detections", []),
                    tracked_objects,
                    current_time=sim_clock,
                )
                track_pose_map = {t["track_id"]: t.get("pose_action", "STANDING (BEKLEME)") for t in tracked_objects}
                for a in alerts:
                    a["pose_action"] = track_pose_map.get(a.get("track_id"), "STANDING (BEKLEME)")
                for ba in bag_alerts:
                    owner_id = ba.get("owner_track_id", 0)
                    owner_trk = next((t for t in tracked_objects if t["track_id"] == owner_id), tracked_objects[0] if tracked_objects else None)
                    last_bag_ts = last_bag_cloud_alert.get(owner_id, 0.0)
                    should_send_bag = (sim_clock - last_bag_ts) >= 20.0
                    if should_send_bag:
                        last_bag_cloud_alert[owner_id] = sim_clock
                    alerts.append(
                        {
                            "message": ba["message"],
                            "track_id": owner_id,
                            "roi_name": f"Abandoned_{ba['class_name']}",
                            "dwell_time": ba["unattended_seconds"],
                            "efficiency": 0.05,
                            "bbox": owner_trk["bbox"] if owner_trk else ba["bbox"],
                            "pose_action": "ABANDONED_OBJECT (TERK EDILMIS NESNE)",
                            "timestamp": sim_clock,
                            "should_send_to_cloud": should_send_bag,
                        }
                    )

            # 5. KVKK Privacy Bridge -> Cloud Layer (ArcFace 512-D + Milvus Watchlist Search + Gemini VLM)
            for alert in alerts:
                if alert.get("should_send_to_cloud"):
                    cloud_res = privacy_bridge.dispatch_suspicious_event(clean_frame_for_cloud, alert)
                    if cloud_res.get("watchlist_hit") and cloud_res.get("best_match"):
                        watchlist_matches[alert["track_id"]] = cloud_res["best_match"]
                        bm = cloud_res["best_match"]
                        print(
                            f"🚨 [CLOUD MILVUS MATCH] Track #{alert['track_id']} in {alert['roi_name']} "
                            f"-> {bm['name']} (Cosine Similarity={bm['cosine_similarity']:.4f}, "
                            f"Risk={bm['risk_level']})"
                        )

            # 6. Annotate Frame for OpenCV GUI & Web Dashboard Stream
            behavior_analyzer.draw_roi(frame)
            draw_detections(frame, tracked_objects, watchlist_matches=watchlist_matches)

            for idx, alert in enumerate(alerts):
                draw_alert(frame, alert["message"], position=(20, 48 + idx * 42))

            inst_fps = 1.0 / max(1e-4, time.perf_counter() - t_frame_start)
            fps = 0.85 * fps + 0.15 * inst_fps

            # Top-right HUD telemetry panel
            hud_x = frame.shape[1] - 375
            cv2.rectangle(frame, (hud_x - 10, 12), (frame.shape[1] - 12, 132), (18, 24, 38), -1)
            cv2.rectangle(frame, (hud_x - 10, 12), (frame.shape[1] - 12, 132), (56, 189, 248), 1)
            cv2.putText(
                frame,
                f"FPS: {fps:.1f} | YOLOv5m ({detector.device_str.upper()})",
                (hud_x, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (74, 222, 128),
                2,
            )
            cv2.putText(
                frame,
                f"Tespit: {len(dets_with_conf)} | DeepSORT Iz: {len(tracked_objects)}",
                (hud_x, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (226, 232, 240),
                1,
            )
            cv2.putText(
                frame,
                f"KVKK Yerel Kare: {privacy_bridge.blocked_raw_frames_count} (Ham Video Korundu)",
                (hud_x, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (56, 189, 248),
                1,
            )
            cv2.putText(
                frame,
                f"Bulut Aktarimi: {privacy_bridge.transmitted_events_count} | Milvus Eslesme: {len(watchlist_matches)}",
                (hud_x, 114),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (251, 191, 36),
                1,
            )

            # Update Web Dashboard stream buffer in Cloud Engine
            ok, jpg_buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                cloud_engine.latest_jpeg_frame = jpg_buf.tobytes()
            cloud_engine.edge_telemetry.update(
                {
                    "fps": round(fps, 1),
                    "active_detections": len(dets_with_conf),
                    "active_tracks": len(tracked_objects),
                    "frames_processed_on_edge": privacy_bridge.blocked_raw_frames_count,
                    "last_update": time.strftime("%H:%M:%S"),
                }
            )

            if can_show_gui:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames > 0 and frame_idx >= args.max_frames:
                print(f"[Edge Layer] Completed {frame_idx} frames (--max-frames reached).")
                break

    except KeyboardInterrupt:
        print("\nKullanıcı tarafından durduruldu.")
    finally:
        shared_det_state["running"] = False
        cap.release()
        if can_show_gui:
            try:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            except Exception:
                pass
        print("Sistem kaynakları temizlendi.")


if __name__ == "__main__":
    main()
