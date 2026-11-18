"""Automated End-to-End Test Suite for TÜBİTAK 2209-A Hybrid Surveillance & Fair Face Recognition System."""

import os
import sys
import cv2
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.helpers import load_config
from utils.demo_video_generator import ensure_demo_assets
from modules.person_detector import PersonDetector
from modules.object_tracker import ObjectTracker
from modules.behavior_analyzer import BehaviorAnalyzer
from modules.face_recognizer import FairFaceRecognizer
from modules.vector_db import MilvusWatchlistDB
from modules.fairness_evaluator import TubitakEvaluationSuite


def test_yolov5_person_detection_on_demo_cctv():
    """Verify YOLOv5m detects both people in the generated demo CCTV video accurately."""
    assets = ensure_demo_assets()
    cap = cv2.VideoCapture(assets["video_path"])
    ret, frame = cap.read()
    cap.release()
    assert ret and frame is not None

    detector = PersonDetector(model_name="yolov5m.pt", confidence_threshold=0.40)
    dets = detector.detect_with_conf(frame)
    assert len(dets) >= 2, f"Expected at least 2 detected persons, got {len(dets)}"
    for xmin, ymin, xmax, ymax, conf in dets:
        assert xmax > xmin and ymax > ymin
        assert 0.40 <= conf <= 1.0


def test_deepsort_tracker_maintains_consistent_ids():
    """Verify DeepSORT (Kalman + Re-ID) maintains consistent track IDs across frames."""
    tracker = ObjectTracker(max_disappeared=15, max_distance=140.0, min_hits=1)
    frame = np.full((720, 1280, 3), 120, dtype=np.uint8)

    # Person A near (300, 380) and Person B moving from (900, 390)
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


def test_loitering_behavior_dual_criteria():
    """Verify BehaviorAnalyzer triggers ONLY when both Temporal and Trajectory Efficiency criteria are met."""
    config = load_config()
    analyzer = BehaviorAnalyzer(config["loitering_analysis"], demo_mode=False)

    # Track 1: Loiters inside Main_Entrance_ROI ([80,140]-[560,620]) for > 180s with pacing motion (Eff < 0.3)
    # Track 2: Walks linearly through Sensitive_Area_ROI ([660,160]-[1180,620]) (Eff > 0.85)
    alerts = []
    for step in range(25):
        t_sec = 1000.0 + step * 9.0  # reaches 216s (> 180s)
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
    assert 10 in alert_track_ids, "Loitering suspect (Track 10) must trigger an alert"
    assert 20 not in alert_track_ids, "Linearly walking person (Track 20) must not trigger false alert"
    assert isinstance(alerts[0]["message"], str)


def test_arcface_512d_and_milvus_watchlist_search(tmp_path):
    """Verify 512-D ArcFace embedding extraction and Cosine Similarity Watchlist matching."""
    assets = ensure_demo_assets()
    img1 = cv2.imread(assets["suspect1_image"])
    img2 = cv2.imread(assets["suspect2_image"])

    recognizer = FairFaceRecognizer(embedding_dim=512, match_threshold=0.52)
    aligned1, _, _ = recognizer.detect_and_align_face(img1)
    emb1 = recognizer.extract_embedding(aligned1, already_aligned=True)

    aligned2, _, _ = recognizer.detect_and_align_face(img2)
    emb2 = recognizer.extract_embedding(aligned2, already_aligned=True)

    assert emb1.shape == (512,)
    assert np.isclose(np.linalg.norm(emb1), 1.0, atol=1e-4)

    db = MilvusWatchlistDB(
        collection_name="test_tubitak_watchlist",
        local_fallback_path=str(tmp_path / "test_wl.json"),
    )
    db.enroll_identity(
        person_id="TEST-WL-1",
        name="Test Suspect 1",
        embedding=emb1,
        risk_level="CRITICAL",
        demographic_group="Middle_Eastern",
    )

    hit_res = db.search_watchlist(emb1, threshold=0.52)
    assert hit_res["is_watchlist_hit"] is True
    assert hit_res["best_match"]["person_id"] == "TEST-WL-1"
    assert hit_res["best_match"]["cosine_similarity"] > 0.95


def test_all_four_tubitak_targets_pass():
    """Verify Hedef 1, Hedef 2, Hedef 3, and Hedef 4 meet or exceed TÜBİTAK 2209-A thresholds."""
    config = load_config()
    suite = TubitakEvaluationSuite(config)
    report = suite.run_all_benchmarks()

    assert report["all_targets_passed"] is True
    assert report["hedef_1_performance"]["avg_fps_per_stream"] >= 15.0
    assert report["hedef_2_behavior"]["f1_score"] >= 0.80
    assert report["hedef_3_face_recognition"]["fairface_frr_percent"] < 5.0
    assert report["hedef_4_fairness"]["disparity_reduction_percent"] >= 25.0
