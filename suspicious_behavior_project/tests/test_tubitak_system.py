"""TÜBİTAK 2209-A Hibrit Gözetim, Risk Skorlama, Alarm ve Adil Yüz Tanıma Otomatik Test Paketi."""

import os
import sys
import time
import cv2
import numpy as np

PROJE_KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJE_KOK not in sys.path:
    sys.path.insert(0, PROJE_KOK)

from utils.helpers import load_config
from utils.demo_video_generator import ensure_demo_assets
from modules.person_detector import PersonDetector
from modules.object_tracker import ObjectTracker
from modules.behavior_analyzer import BehaviorAnalyzer
from modules.face_recognizer import FairFaceRecognizer
from modules.vector_db import MilvusWatchlistDB
from modules.fairness_evaluator import TubitakEvaluationSuite
from modules.smart_analytics import SmartIntelligenceEngine
from modules.risk_scorer import PersonRiskScorer
from modules.alarm_manager import AlarmSystemManager


def test_yolov5_kisi_tespiti_ornek_video():
    """YOLOv5m modelinin örnek CCTV test videosundaki kişileri doğru tespit ettiğini doğrular."""
    assets = ensure_demo_assets()
    cap = cv2.VideoCapture(assets["video_path"])
    ret, frame = cap.read()
    cap.release()
    assert ret and frame is not None

    detector = PersonDetector(model_name="yolov5m.pt", confidence_threshold=0.40)
    dets = detector.detect_with_conf(frame)
    assert len(dets) >= 2, f"En az 2 kişi tespiti bekleniyordu, bulunan: {len(dets)}"
    for xmin, ymin, xmax, ymax, conf in dets:
        assert xmax > xmin and ymax > ymin
        assert 0.40 <= conf <= 1.0


def test_deepsort_takip_kimlik_tutarliligi():
    """DeepSORT (Kalman + Re-ID) modülünün kareler arasında tutarlı iz kimliği koruduğunu doğrular."""
    tracker = ObjectTracker(max_disappeared=15, max_distance=140.0, min_hits=1)
    frame = np.full((720, 1280, 3), 120, dtype=np.uint8)

    for step in range(8):
        rects = [
            (250 + step * 2, 230, 360 + step * 2, 530),
            (850 - step * 8, 240, 960 - step * 8, 540),
        ]
        tracks = tracker.update(rects, frame=frame)

    assert len(tracks) == 2
    ids = sorted([t["track_id"] for t in tracks])
    assert ids == [0, 1]
    assert len(tracks[0]["history"]) == 8


def test_amacsis_bekleme_cift_kriter_mantigi():
    """BehaviorAnalyzer modülünün yalnızca Zamansal ve Yörüngesel kriter birlikte sağlanınca uyardığını doğrular."""
    config = load_config()
    analyzer = BehaviorAnalyzer(config["loitering_analysis"], demo_mode=False)

    alerts = []
    for step in range(25):
        t_sec = 1000.0 + step * 9.0
        angle = step * 0.7
        lx = int(300 + 30 * np.cos(angle))
        ly = int(380 + 30 * np.sin(angle))

        wx = int(680 + step * 18)
        wy = 380

        tracks = [
            {"track_id": 10, "bbox": [lx - 40, ly - 100, lx + 40, ly + 100]},
            {"track_id": 20, "bbox": [wx - 40, wy - 100, wx + 40, wy + 100]},
        ]
        alerts = analyzer.update(tracks, current_timestamp=t_sec)

    alert_track_ids = [a["track_id"] for a in alerts]
    assert 10 in alert_track_ids, "Oyalanan şüpheli (İz 10) uyarı üretmelidir"
    assert 20 not in alert_track_ids, "Doğrusal yürüyen yaya (İz 20) yanlış alarm üretmemelidir"
    assert isinstance(alerts[0]["message"], str)


def test_arcface_512d_ve_milvus_izleme_listesi(tmp_path):
    """512-D ArcFace gömme çıkarımı ve Milvus Kosinüs Benzerliği eşleşmesini doğrular."""
    assets = ensure_demo_assets()
    img1 = cv2.imread(assets["suspect1_image"])

    recognizer = FairFaceRecognizer(embedding_dim=512, match_threshold=0.52)
    aligned1, _, _ = recognizer.detect_and_align_face(img1)
    emb1 = recognizer.extract_embedding(aligned1, already_aligned=True)

    assert emb1.shape == (512,)
    assert np.isclose(np.linalg.norm(emb1), 1.0, atol=1e-4)

    db = MilvusWatchlistDB(
        collection_name="test_tubitak_watchlist",
        local_fallback_path=str(tmp_path / "test_wl.json"),
    )
    db.enroll_identity(
        person_id="TEST-WL-1",
        name="Test Şüphelisi 1",
        embedding=emb1,
        risk_level="KRİTİK",
        demographic_group="Orta_Dogulu",
    )

    hit_res = db.search_watchlist(emb1, threshold=0.52)
    assert hit_res["is_watchlist_hit"] is True
    assert hit_res["best_match"]["person_id"] == "TEST-WL-1"
    assert hit_res["best_match"]["cosine_similarity"] > 0.95


def test_dort_tubitak_hedefi_dogrulama():
    """Hedef 1, Hedef 2, Hedef 3 ve Hedef 4 ölçütlerinin tamamının karşılandığını doğrular."""
    config = load_config()
    suite = TubitakEvaluationSuite(config)
    report = suite.run_all_benchmarks()

    assert report["all_targets_passed"] is True
    assert report["hedef_1_performance"]["avg_fps_per_stream"] >= 15.0
    assert report["hedef_2_behavior"]["f1_score"] >= 0.80
    assert report["hedef_3_face_recognition"]["fairface_frr_percent"] < 5.0
    assert report["hedef_4_fairness"]["disparity_reduction_percent"] >= 25.0


def test_alti_akilli_yapay_zeka_modulu():
    """6 akıllı yapay zeka uzantısının (KVKK, Tahmin, Poz, Çanta, Canlılık, Adli Arama) çalıştığını doğrular."""
    engine = SmartIntelligenceEngine(width=1280, height=720)
    engine.heatmap_enabled = True
    engine.trajectory_prediction_enabled = True
    engine.pose_estimation_enabled = True

    frame = np.full((720, 1280, 3), 90, dtype=np.uint8)
    tracks = [
        {
            "track_id": 1,
            "bbox": [200, 180, 310, 500],
            "centroid": (255, 340),
            "velocity": (4.0, 1.0),
            "history": [(235, 335), (240, 336), (248, 338), (255, 340)],
            "behavior_status": "NORMAL",
        },
        {
            "track_id": 2,
            "bbox": [600, 380, 820, 490],  # Yatay dikdörtgen -> DÜŞME TESPİTİ
            "centroid": (710, 435),
            "velocity": (0.0, 0.0),
            "history": [(710, 435)],
            "behavior_status": "ŞÜPHELİ OYALANMA",
        },
    ]
    rois = [{"name": "Test_ROI", "points": [[280, 100], [580, 100], [580, 600], [280, 600]]}]

    masked_count = engine.apply_dynamic_kvkk_masking(frame, tracks)
    assert masked_count == 1

    warnings = engine.update_and_draw_heatmap_and_predictions(frame, tracks, rois)
    assert "predicted_point" in tracks[0]
    assert len(warnings) >= 1

    pose_alerts = engine.analyze_and_draw_pose_actions(frame, tracks)
    assert any("DÜŞME TESPİTİ" in pa["action"] for pa in pose_alerts)

    engine.trigger_simulated_abandoned_bag()
    bag_alerts = engine.update_and_draw_abandoned_objects(
        frame, [], tracks, current_time=time.time()
    )
    assert len(bag_alerts) >= 1

    assets = ensure_demo_assets()
    face_img = cv2.resize(cv2.imread(assets["suspect1_image"]), (112, 112))
    liveness = engine.evaluate_face_liveness(face_img)
    assert 0.15 <= liveness["liveness_score"] <= 0.99

    clothing = engine.extract_clothing_attributes(face_img)
    report = engine.generate_gemini_vlm_security_report(
        face_img,
        {
            "camera_id": "KAM-01",
            "roi_name": "Ana_Giris_Bolgesi",
            "dwell_time": 6.5,
            "efficiency": 0.18,
            "pose_action": "BEKLEME",
            "risk_percent": 78.0,
            "risk_tier": "YÜKSEK",
        },
        liveness,
        clothing,
    )
    assert "YZ Analizi:" in report

    sample_events = [
        {"event_id": "OLY-1", "upper_clothing": "Siyah", "roi_name": "Ana_Giris_Bolgesi", "behavior_summary": report},
        {"event_id": "OLY-2", "upper_clothing": "Beyaz", "roi_name": "Hassas_Bolge", "behavior_summary": "Normal"},
    ]
    found = engine.search_events_by_query(sample_events, "siyah ana_giris")
    assert len(found) == 1 and found[0]["event_id"] == "OLY-1"


def test_kisi_risk_yuzdesi_hesaplama_sistemi():
    """PersonRiskScorer modülünün DÜŞÜK ve KRİTİK senaryolarda doğru risk yüzdesi ürettiğini doğrular."""
    scorer = PersonRiskScorer()

    low_res = scorer.compute_person_risk(
        behavior_status="NORMAL",
        roi_name=None,
        dwell_time=0.0,
        efficiency=0.95,
        pose_action="YÜRÜME",
        watchlist_hit=False,
        liveness_is_live=True,
    )
    assert low_res["risk_percent"] < 30.0
    assert low_res["risk_tier"] == "DÜŞÜK"

    crit_res = scorer.compute_person_risk(
        behavior_status="ŞÜPHELİ OYALANMA",
        roi_name="Hassas_Bolge",
        dwell_time=12.0,
        time_threshold=5.0,
        efficiency=0.10,
        pose_action="EĞİLME / SIZMA",
        watchlist_hit=True,
        watchlist_risk_level="KRİTİK",
        liveness_is_live=False,
    )
    assert crit_res["risk_percent"] >= 80.0
    assert crit_res["risk_tier"] == "KRİTİK"


def test_cok_seviyeli_alarm_ve_ihbar_sistemi(tmp_path):
    """AlarmSystemManager modülünün alarm üretme, onaylama ve çözme yaşam döngüsünü doğrular."""
    mgr = AlarmSystemManager(log_path=str(tmp_path / "test_alarms.json"), cooldown_seconds=10.0)

    alm = mgr.evaluate_and_trigger(
        title="Hassas Bölgede Kritik İhlal",
        description="Test kritik alarm kaydı",
        category="MİLVUS EŞLEŞMESİ (KRİTİK)",
        risk_percent=91.5,
        camera_id="KAM-01",
        track_id=3,
        roi_name="Hassas_Bolge",
    )
    assert alm is not None
    assert alm["seviye"] == "KRİTİK"
    assert alm["durum"] == "AKTİF"
    assert mgr.get_summary()["aktif_alarm_sayisi"] == 1

    acked = mgr.acknowledge_alarm(alm["alarm_id"])
    assert acked["durum"] == "ONAYLANDI"

    resolved = mgr.resolve_alarm(alm["alarm_id"])
    assert resolved["durum"] == "ÇÖZÜLDÜ"
    assert mgr.get_summary()["aktif_alarm_sayisi"] == 0
