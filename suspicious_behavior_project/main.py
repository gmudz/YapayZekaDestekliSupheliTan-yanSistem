"""TÜBİTAK 2209-A Hibrit Uç/Bulut Şüpheli Davranış, Risk Skorlama ve Adil Yüz Tanıma Ana Uygulaması.

Bütünleştirdiği bileşenler:
1. Uç Katmanı:
   - YOLOv5m Gerçek Zamanlı Kişi ve Çanta Tespiti (modules/person_detector.py)
   - DeepSORT Çoklu Nesne Takibi: 8-D Kalman Filtresi + 512-D Re-ID (modules/object_tracker.py)
   - Çift Kriterli Amaçsız Bekleme Analizi (modules/behavior_analyzer.py)
   - 6 Akıllı Yapay Zeka Modülü: KVKK Maskeleme, Isı Haritası, 3sn Tahmin, İskelet Pozu, Terk Edilmiş Nesne
   - Kişi Risk Yüzdesi Hesaplama Motoru %0-%100 (modules/risk_scorer.py)
   - KVKK 6698 Veri Minimizasyonu Köprüsü (modules/privacy_bridge.py)
2. Bulut Katmanı:
   - FastAPI Bulut Servisi, Alarm Yönetim Merkezi ve Google Stitch Web Paneli (:8000)
   - FairFace Kalibrasyonlu 512-D ArcFace Yüz Tanıma (modules/face_recognizer.py)
   - Milvus Standalone Vektör Veritabanı Kosinüs Benzerliği Araması (modules/vector_db.py)
"""

import os
import time
import argparse
import threading
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from modules.person_detector import PersonDetector
from modules.object_tracker import ObjectTracker
from modules.behavior_analyzer import BehaviorAnalyzer
from modules.privacy_bridge import KVKKPrivacyBridge
from modules.fairness_evaluator import TubitakEvaluationSuite
from modules.risk_scorer import PersonRiskScorer
from utils.helpers import load_config, load_camera_source, draw_alert, ascii_tr


def convert_detections(
    detections: List[Tuple[int, int, int, int, float]]
) -> Tuple[List[List[float]], List[float]]:
    """Tespitleri (xmin, ymin, xmax, ymax, güven) biçiminden ([x, y, w, h], güven) biçimine çevirir."""
    xywh_detections = []
    confidences = []
    for det in detections:
        xmin, ymin, xmax, ymax = det[:4]
        conf = det[4] if len(det) >= 5 else 0.90
        xywh_detections.append([xmin, ymin, xmax - xmin, ymax - ymin])
        confidences.append(conf)
    return xywh_detections, confidences


def draw_detections(
    frame: np.ndarray,
    tracked_objects: List[Dict[str, Any]],
    watchlist_matches: Optional[Dict[int, Dict[str, Any]]] = None,
) -> None:
    """Takip edilen kişileri, yörünge izlerini, anlık risk yüzdesini (%0-%100) ve Milvus eşleşmelerini çizer."""
    watchlist_matches = watchlist_matches or {}

    for obj in tracked_objects:
        bbox = obj["bbox"]
        track_id = obj["track_id"]
        status = obj.get("behavior_status", "NORMAL")
        dwell = obj.get("dwell_time", 0.0)
        eff = obj.get("efficiency", 1.0)
        risk_pct = float(obj.get("risk_percent", 0.0))
        risk_tier = obj.get("risk_tier", "DÜŞÜK")
        color = PersonRiskScorer.TIER_COLORS_BGR.get(risk_tier, (74, 222, 128))

        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 4, color, -1)

        history = obj.get("history", [])
        if len(history) > 1:
            pts = history[-40:]
            for i in range(1, len(pts)):
                if pts[i] is not None and pts[i - 1] is not None:
                    cv2.line(frame, pts[i - 1], pts[i], color, 2)

        # Üst başlık: İz Kimliği + Risk Yüzdesi + Seviye
        header = ascii_tr(f"Iz:{track_id} | Risk:%{risk_pct:.0f} [{risk_tier}]")
        cv2.putText(
            frame,
            header,
            (x1, max(20, y1 - 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
        )

        # Mini Risk Çubuğu
        bar_w = max(70, min(180, x2 - x1))
        bar_y1 = max(24, y1 - 24)
        cv2.rectangle(frame, (x1, bar_y1), (x1 + bar_w, bar_y1 + 6), (30, 35, 48), -1)
        fill_w = int(bar_w * (risk_pct / 100.0))
        if fill_w > 0:
            cv2.rectangle(frame, (x1, bar_y1), (x1 + fill_w, bar_y1 + 6), color, -1)

        telemetry = ascii_tr(f"{status} | Sure:{dwell:.1f}s | Verim:{eff:.2f}")
        cv2.putText(
            frame,
            telemetry,
            (x1, max(40, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (235, 245, 255),
            1,
        )

        if track_id in watchlist_matches:
            match_info = watchlist_matches[track_id]
            match_text = ascii_tr(
                f"MILVUS ESLESME: {match_info['name']} (Benzerlik:%{match_info['cosine_similarity']*100:.1f})"
            )
            cv2.rectangle(
                frame,
                (x1, min(frame.shape[0] - 28, y2 + 4)),
                (min(frame.shape[1] - 5, x1 + 420), min(frame.shape[0] - 2, y2 + 28)),
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
    """FastAPI Bulut Katmanı ve Web Panelini arka plan iş parçacığında başlatır."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            print(f"[Bulut Katmanı] FastAPI Web Paneli zaten çalışıyor: http://127.0.0.1:{port}")
            from cloud.server import cloud_engine

            return cloud_engine

    import uvicorn
    from cloud.server import app, cloud_engine

    def _run():
        uvicorn.run(app, host=host, port=port, log_level="warning")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    time.sleep(1.0)
    print(f"[Bulut Katmanı] FastAPI ve Web Paneli başlatıldı: http://127.0.0.1:{port}")
    return cloud_engine


def print_evaluation_report(report: Dict[str, Any]) -> None:
    """TÜBİTAK 2209-A Hedef 1-4 doğrulama raporunu konsola yazdırır."""
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
        f"   -> Kesinlik: {h2['precision']} | Duyarlılık: {h2['recall']} | "
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
        description="TÜBİTAK 2209-A Hibrit Şüpheli Davranış, Risk Skorlama ve Adil Yüz Tanıma Sistemi"
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
        help="OpenCV penceresi açmadan yalnızca Web Gösterge Paneli ile çalıştır",
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

    cloud_cfg = config.get("cloud", {})
    cloud_engine = start_cloud_server_in_background(
        host=cloud_cfg.get("host", "0.0.0.0"),
        port=int(cloud_cfg.get("port", 8000)),
    )

    if cloud_engine.latest_benchmark:
        print_evaluation_report(cloud_engine.latest_benchmark)

    cap = load_camera_source(config)

    detector = PersonDetector(
        model_name=config["model"]["weights"],
        confidence_threshold=config["model"]["confidence_threshold"],
        iou_threshold=config["model"].get("iou_threshold", 0.45),
        device=config["model"].get("device", "auto"),
        half_precision=config["model"].get("half_precision", False),
        input_size=config["model"].get("input_size", 320),
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
        camera_id=config["camera"].get("camera_id", "KAM-01-GİRİŞ"),
        enabled=True,
        local_cloud_fallback=cloud_engine,
    )

    can_show_gui = (not args.headless) and bool(
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )
    window_name = "TUBITAK 2209-A | Akilli Guvenlik, Risk ve Adil Yuz Tanima Sistemi"
    if can_show_gui:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720)
        except Exception:
            can_show_gui = False

    print(
        f"[Uç Katmanı] İşlem hattı başlatıldı (YOLOv5m [{detector.device_str}] + DeepSORT + "
        f"Risk Motoru + Alarm Sistemi + Bekleme Eşiği={behavior_analyzer.time_threshold}sn)"
    )
    print(f"[Web Paneli] Tarayıcıdan erişin: http://127.0.0.1:{cloud_cfg.get('port', 8000)}")

    fps = 25.0
    frame_idx = 0
    watchlist_matches: Dict[int, Dict[str, Any]] = {}
    last_bag_cloud_alert: Dict[int, float] = {}
    sim_clock = time.time()
    live_webcam_enrolled = False

    det_lock = threading.Lock()
    shared_det_state: Dict[str, Any] = {
        "frame_to_process": None,
        "latest_dets": [],
        "latest_bags": [],
        "running": True,
    }

    def _async_detector_worker():
        while shared_det_state["running"]:
            frm = shared_det_state["frame_to_process"]
            if frm is not None:
                shared_det_state["frame_to_process"] = None
                res = detector.detect_with_conf(frm)
                bags = list(getattr(detector, "last_bag_detections", []))
                with det_lock:
                    shared_det_state["latest_dets"] = res
                    shared_det_state["latest_bags"] = bags
            else:
                time.sleep(0.01)

    det_thread = threading.Thread(target=_async_detector_worker, daemon=True)
    det_thread.start()

    try:
        while True:
            t_frame_start = time.perf_counter()

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
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Video akışı sona erdi.")
                    break

            if frame.shape[1] != 1280 or frame.shape[0] != 720:
                frame = cv2.resize(frame, (1280, 720))

            frame_idx += 1
            privacy_bridge.register_frame_processed_on_edge()

            is_live_webcam = str(config["camera"].get("source", "0")) in ("0", "/dev/video0")
            if is_live_webcam or not demo_mode:
                sim_clock = time.time()
            else:
                sim_clock += 0.20

            # 1. Uç Katmanı Kişi ve Çanta Tespiti (YOLOv5m)
            if is_live_webcam and args.max_frames == 0:
                if shared_det_state["frame_to_process"] is None:
                    shared_det_state["frame_to_process"] = frame.copy()
                with det_lock:
                    dets_with_conf = list(shared_det_state["latest_dets"])
                    bag_dets = list(shared_det_state["latest_bags"])
                if frame_idx == 1 and not dets_with_conf:
                    dets_with_conf = detector.detect_with_conf(frame)
                    bag_dets = list(getattr(detector, "last_bag_detections", []))
                    with det_lock:
                        shared_det_state["latest_dets"] = dets_with_conf
                        shared_det_state["latest_bags"] = bag_dets
            else:
                dets_with_conf = detector.detect_with_conf(frame)
                bag_dets = list(getattr(detector, "last_bag_detections", []))

            det_boxes = [(d[0], d[1], d[2], d[3]) for d in dets_with_conf]
            det_confs = [d[4] for d in dets_with_conf]

            # 2. Uç Katmanı Çoklu Nesne Takibi (DeepSORT)
            tracked_objects = tracker.update(
                det_boxes, frame=frame, confidences=det_confs
            )

            # Canlı web kamerasında ilk yüz tespitinde otomatik Milvus kaydı
            if is_live_webcam and not live_webcam_enrolled and tracked_objects:
                try:
                    trk0 = tracked_objects[0]
                    crop0 = privacy_bridge.crop_suspect_region(frame, trk0["bbox"])
                    aligned0, _, _ = cloud_engine.recognizer.detect_and_align_face(crop0)
                    emb0 = cloud_engine.recognizer.extract_embedding(aligned0, already_aligned=True)
                    b64_0 = privacy_bridge.encode_image_b64(aligned0)
                    cloud_engine.vector_db.enroll_identity(
                        person_id="WL-CANLI-KAM",
                        name="Canlı Kamera Kullanıcısı (Aktif İzleme)",
                        embedding=emb0,
                        risk_level="KRİTİK",
                        demographic_group="Orta_Dogulu",
                        image_b64=b64_0,
                        notes="Dizüstü bilgisayar kamerasından otomatik izleme listesi kaydı",
                    )
                    live_webcam_enrolled = True
                    print("[Canlı Kamera] Yüz profili Milvus İzleme Listesine (WL-CANLI-KAM) kaydedildi.")
                except Exception as e:
                    print(f"[Canlı Kamera] Otomatik kayıt uyarısı: {e}")

            # 3. Uç Katmanı Davranış Analizi
            alerts = behavior_analyzer.update(
                tracked_objects, current_timestamp=sim_clock
            )

            if ctrl.get("force_trigger_alert") and tracked_objects:
                ctrl["force_trigger_alert"] = False
                target_trk = tracked_objects[0]
                target_trk["behavior_status"] = "ŞÜPHELİ OYALANMA"
                alerts.append(
                    {
                        "message": f"UYARI: Kişi ID {target_trk['track_id']} - Ana_Giris_Bolgesi (Manuel Test)",
                        "track_id": target_trk["track_id"],
                        "roi_name": target_trk.get("roi_name") or "Ana_Giris_Bolgesi",
                        "dwell_time": max(behavior_analyzer.time_threshold + 2.5, target_trk.get("dwell_time", 6.5)),
                        "efficiency": 0.14,
                        "bbox": target_trk["bbox"],
                        "centroid": target_trk["centroid"],
                        "timestamp": sim_clock,
                        "should_send_to_cloud": True,
                    }
                )

            # Bulut kırpımı için temiz kareyi sakla
            clean_frame_for_cloud = frame.copy()

            # 4. Akıllı YZ Modülleri ve Kişi Risk Yüzdesi Hesaplaması
            smart_engine = getattr(cloud_engine, "smart_engine", None)
            bag_alerts: List[Dict[str, Any]] = []
            if smart_engine is not None:
                smart_engine.apply_dynamic_kvkk_masking(frame, tracked_objects)
                smart_engine.update_and_draw_heatmap_and_predictions(
                    frame, tracked_objects, behavior_analyzer.rois
                )
                pose_alerts = smart_engine.analyze_and_draw_pose_actions(frame, tracked_objects)
                bag_alerts = smart_engine.update_and_draw_abandoned_objects(
                    frame,
                    bag_dets,
                    tracked_objects,
                    current_time=sim_clock,
                )

                # Poz anomalisi varsa alarm sistemini bilgilendir
                for pa in pose_alerts:
                    if "DÜŞME" in pa["action"]:
                        cloud_engine.alarm_manager.evaluate_and_trigger(
                            title=f"Acil Vücut Pozu Anomalisi: {pa['action']}",
                            description=f"İz #{pa['track_id']} için acil düşme/tehlike pozisyonu tespit edildi.",
                            category="DÜŞME / ACİL DURUM",
                            risk_percent=92.0,
                            camera_id=privacy_bridge.camera_id,
                            track_id=pa["track_id"],
                        )

            abandoned_owners = {ba.get("owner_track_id") for ba in bag_alerts}
            active_risk_list: List[Dict[str, Any]] = []

            for obj in tracked_objects:
                tid = obj["track_id"]
                bm = watchlist_matches.get(tid)
                risk_res = cloud_engine.risk_scorer.compute_person_risk(
                    behavior_status=obj.get("behavior_status", "NORMAL"),
                    roi_name=obj.get("roi_name"),
                    dwell_time=obj.get("dwell_time", 0.0),
                    time_threshold=behavior_analyzer.time_threshold,
                    efficiency=obj.get("efficiency", 1.0),
                    pose_action=obj.get("pose_action", "BEKLEME"),
                    watchlist_hit=bool(bm),
                    watchlist_risk_level=(bm or {}).get("risk_level"),
                    liveness_is_live=True,
                    pre_breach=bool(obj.get("pre_breach", False)),
                    abandoned_bag_owner=(tid in abandoned_owners),
                )
                obj["risk_percent"] = risk_res["risk_percent"]
                obj["risk_tier"] = risk_res["risk_tier"]
                obj["risk_breakdown"] = risk_res["breakdown"]
                active_risk_list.append(
                    {
                        "track_id": tid,
                        "risk_percent": risk_res["risk_percent"],
                        "risk_tier": risk_res["risk_tier"],
                        "behavior_status": obj.get("behavior_status", "NORMAL"),
                        "roi_name": obj.get("roi_name") or "Bölge Dışı",
                        "pose_action": obj.get("pose_action", "BEKLEME"),
                        "dwell_time": obj.get("dwell_time", 0.0),
                        "efficiency": obj.get("efficiency", 1.0),
                        "breakdown": risk_res["breakdown"],
                    }
                )

                # Erken ihlal uyarısı veya yüksek risk için anlık alarm kontrolü
                if obj.get("pre_breach"):
                    cloud_engine.alarm_manager.evaluate_and_trigger(
                        title=f"Erken Bölge İhlali Tahmini (İz #{tid})",
                        description=f"İz #{tid} 3 saniye içinde kritik bölgeye giriş yapacak şekilde ilerliyor.",
                        category="ERKEN İHLAL",
                        risk_percent=max(38.0, risk_res["risk_percent"]),
                        camera_id=privacy_bridge.camera_id,
                        track_id=tid,
                    )

            cloud_engine.active_person_risks = sorted(
                active_risk_list, key=lambda x: x["risk_percent"], reverse=True
            )
            max_risk_now = (
                max([r["risk_percent"] for r in active_risk_list]) if active_risk_list else 0.0
            )

            track_map = {t["track_id"]: t for t in tracked_objects}
            for a in alerts:
                trk = track_map.get(a.get("track_id"), {})
                a["pose_action"] = trk.get("pose_action", "BEKLEME")
                a["risk_percent"] = trk.get("risk_percent", 0.0)

            for ba in bag_alerts:
                owner_id = ba.get("owner_track_id", 0)
                owner_trk = track_map.get(owner_id, tracked_objects[0] if tracked_objects else None)
                last_bag_ts = last_bag_cloud_alert.get(owner_id, 0.0)
                should_send_bag = (sim_clock - last_bag_ts) >= 20.0
                if should_send_bag:
                    last_bag_cloud_alert[owner_id] = sim_clock
                alerts.append(
                    {
                        "message": ba["message"],
                        "track_id": owner_id,
                        "roi_name": f"Terk_Edilmis_{ba['class_name']}",
                        "dwell_time": ba["unattended_seconds"],
                        "efficiency": 0.05,
                        "bbox": owner_trk["bbox"] if owner_trk else ba["bbox"],
                        "pose_action": "TERK EDİLMİŞ NESNE",
                        "risk_percent": owner_trk.get("risk_percent", 85.0) if owner_trk else 85.0,
                        "timestamp": sim_clock,
                        "should_send_to_cloud": should_send_bag,
                    }
                )

            # 5. KVKK Gizlilik Köprüsü -> Bulut Katmanı (ArcFace 512-D + Milvus + Alarm Sistemi)
            for alert in alerts:
                if alert.get("should_send_to_cloud"):
                    cloud_res = privacy_bridge.dispatch_suspicious_event(clean_frame_for_cloud, alert)
                    if cloud_res.get("watchlist_hit") and cloud_res.get("best_match"):
                        watchlist_matches[alert["track_id"]] = cloud_res["best_match"]
                        bm = cloud_res["best_match"]
                        print(
                            f"🚨 [BULUT MİLVUS EŞLEŞMESİ] İz #{alert['track_id']} | Bölge: {alert['roi_name']} "
                            f"-> {bm['name']} (Benzerlik={bm['cosine_similarity']:.4f}, "
                            f"Risk=%{cloud_res.get('risk_percent', 0)})"
                        )

            # 6. Görselleştirme: ROI Bölgeleri, Takip Kutuları, Risk Çubukları ve Acil Alarm Çerçevesi
            behavior_analyzer.draw_roi(frame)
            draw_detections(frame, tracked_objects, watchlist_matches=watchlist_matches)

            for idx, alert in enumerate(alerts):
                draw_alert(frame, alert["message"], position=(20, 48 + idx * 42))

            alarm_summary = cloud_engine.alarm_manager.get_summary()
            if alarm_summary.get("kritik_aktif_sayisi", 0) > 0 and int(time.time() * 3) % 2 == 0:
                cv2.rectangle(frame, (4, 4), (frame.shape[1] - 5, frame.shape[0] - 5), (0, 0, 255), 6)

            inst_fps = 1.0 / max(1e-4, time.perf_counter() - t_frame_start)
            fps = 0.85 * fps + 0.15 * inst_fps

            hud_x = frame.shape[1] - 390
            cv2.rectangle(frame, (hud_x - 10, 12), (frame.shape[1] - 12, 152), (18, 24, 38), -1)
            cv2.rectangle(frame, (hud_x - 10, 12), (frame.shape[1] - 12, 152), (56, 189, 248), 1)
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
                0.50,
                (226, 232, 240),
                1,
            )
            cv2.putText(
                frame,
                f"KVKK Yerel Kare: {privacy_bridge.blocked_raw_frames_count}",
                (hud_x, 86),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (56, 189, 248),
                1,
            )
            cv2.putText(
                frame,
                f"En Yuksek Risk: %{max_risk_now:.0f} | Aktif Alarm: {alarm_summary.get('aktif_alarm_sayisi', 0)}",
                (hud_x, 112),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 255) if max_risk_now >= 55 else (251, 191, 36),
                1,
            )
            cv2.putText(
                frame,
                f"Bulut Aktarimi: {privacy_bridge.transmitted_events_count} | Milvus: {len(watchlist_matches)}",
                (hud_x, 138),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (78, 222, 163),
                1,
            )

            ok, jpg_buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                cloud_engine.latest_jpeg_frame = jpg_buf.tobytes()
            cloud_engine.edge_telemetry.update(
                {
                    "fps": round(fps, 1),
                    "active_detections": len(dets_with_conf),
                    "active_tracks": len(tracked_objects),
                    "frames_processed_on_edge": privacy_bridge.blocked_raw_frames_count,
                    "max_active_risk_percent": round(max_risk_now, 1),
                    "last_update": time.strftime("%H:%M:%S"),
                }
            )

            if can_show_gui:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames > 0 and frame_idx >= args.max_frames:
                print(f"[Uç Katmanı] {frame_idx} kare işlendi (--max-frames sınırına ulaşıldı).")
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
