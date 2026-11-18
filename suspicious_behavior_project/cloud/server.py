"""FastAPI Cloud Layer Service & Real-Time Interactive Web Dashboard (TÜBİTAK 2209-A).

Provides:
- KVKK-compliant Cloud endpoint (/api/v1/events/analyze) receiving ONLY cropped
  suspect images + loitering metadata from the Edge Layer
- 512-D ArcFace Face Embedding extraction with FairFace demographic calibration
- Milvus Vector Database Watchlist search using Cosine Similarity
- Interactive controls (/api/v1/controls) to switch video source (Webcam / Demo CCTV),
  adjust loitering thresholds on the fly, trigger instant test alerts, and test custom
  uploaded images or browser webcam snapshots (/api/v1/test/image)
- Live MJPEG annotated video stream (/api/v1/stream/video) and Web Dashboard (/).
"""

import os
import sys
import time
import base64
import threading
import cv2
import numpy as np
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.helpers import load_config
from utils.demo_video_generator import ensure_demo_assets
from modules.face_recognizer import FairFaceRecognizer
from modules.vector_db import MilvusWatchlistDB
from modules.privacy_bridge import KVKKPrivacyBridge
from modules.fairness_evaluator import TubitakEvaluationSuite


class EdgeEventPayload(BaseModel):
    camera_id: str = "CAM-01-GIRIS"
    track_id: int
    roi_name: str
    dwell_time: float
    efficiency: float
    timestamp: Optional[float] = None
    kvkk_data_minimized: bool = True
    raw_stream_transmitted: bool = False
    cropped_image_b64: str


class RuntimeControlUpdate(BaseModel):
    source: Optional[str] = None
    time_threshold: Optional[float] = None
    efficiency_threshold: Optional[float] = None
    force_trigger_alert: Optional[bool] = None
    clear_events: Optional[bool] = None


class CloudPipelineEngine:
    """Central Cloud Layer orchestrator managing ArcFace, Milvus Watchlist, and Alert History."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        fr_cfg = self.config.get("face_recognition", {})
        mv_cfg = self.config.get("milvus", {})

        self.recognizer = FairFaceRecognizer(
            embedding_dim=int(fr_cfg.get("embedding_dim", 512)),
            match_threshold=float(fr_cfg.get("match_threshold", 0.52)),
            fairness_calibration=bool(fr_cfg.get("fairness_calibration", True)),
        )
        self.vector_db = MilvusWatchlistDB(
            host=mv_cfg.get("host", "127.0.0.1"),
            port=int(mv_cfg.get("port", 19530)),
            collection_name=mv_cfg.get("collection_name", "tubitak_watchlist_faces"),
            embedding_dim=int(mv_cfg.get("embedding_dim", 512)),
            local_fallback_path=mv_cfg.get(
                "local_fallback_path", "data/watchlist_store.json"
            ),
        )
        self.evaluator = TubitakEvaluationSuite(self.config)
        self.events_log: List[Dict[str, Any]] = []
        self.latest_benchmark: Optional[Dict[str, Any]] = None
        self.edge_telemetry: Dict[str, Any] = {
            "fps": 0.0,
            "active_detections": 0,
            "active_tracks": 0,
            "frames_processed_on_edge": 0,
            "active_source": "auto",
            "time_threshold": 5.0,
            "efficiency_threshold": 0.30,
            "camera_id": self.config["camera"].get("camera_id", "CAM-01-GIRIS"),
            "last_update": time.strftime("%H:%M:%S"),
        }
        self.runtime_controls: Dict[str, Any] = {
            "requested_source": None,
            "time_threshold": None,
            "efficiency_threshold": None,
            "force_trigger_alert": False,
        }
        self.latest_jpeg_frame: Optional[bytes] = None
        self.latest_raw_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()

        self._seed_default_watchlist()
        self.latest_benchmark = self.evaluator.run_all_benchmarks()

    def _seed_default_watchlist(self) -> None:
        """Ensure default demo Watchlist identities exist so live demo shows immediate matches."""
        assets = ensure_demo_assets()
        if len(self.vector_db.list_identities()) == 0:
            s1_img = cv2.imread(assets["suspect1_image"])
            if s1_img is not None:
                aligned, _, _ = self.recognizer.detect_and_align_face(s1_img)
                emb = self.recognizer.extract_embedding(aligned, already_aligned=True)
                b64 = KVKKPrivacyBridge.encode_image_b64(aligned)
                self.vector_db.enroll_identity(
                    person_id="WL-2209-01",
                    name="Şüpheli #1 (Kırmızı İzleme Listesi)",
                    embedding=emb,
                    risk_level="CRITICAL",
                    demographic_group="Middle_Eastern",
                    image_b64=b64,
                    notes="Kritik tesis giriş bölgesi (Main_Entrance_ROI) aranan şahıs profili",
                )

    @staticmethod
    def decode_b64_image(b64_str: str) -> np.ndarray:
        if not b64_str:
            return np.zeros((112, 112, 3), dtype=np.uint8)
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        raw = base64.b64decode(b64_str)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return np.zeros((112, 112, 3), dtype=np.uint8)
        return img

    def process_edge_event(
        self, payload: Dict[str, Any], crop_bgr: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Process a KVKK-minimized suspicious behavior event from the Edge Layer."""
        if crop_bgr is None:
            crop_bgr = self.decode_b64_image(payload.get("cropped_image_b64", ""))

        aligned_face, face_bbox, face_conf = self.recognizer.detect_and_align_face(
            crop_bgr
        )
        embedding = self.recognizer.extract_embedding(
            aligned_face, already_aligned=True
        )
        demographic_group = self.recognizer.estimate_demographic_group(aligned_face)

        search_res = self.vector_db.search_watchlist(
            embedding,
            top_k=3,
            threshold=self.recognizer.match_threshold,
        )

        aligned_b64 = KVKKPrivacyBridge.encode_image_b64(aligned_face)
        crop_b64 = payload.get("cropped_image_b64") or KVKKPrivacyBridge.encode_image_b64(
            crop_bgr
        )

        dwell_val = round(float(payload.get("dwell_time", 0.0)), 1)
        eff_val = round(float(payload.get("efficiency", 0.0)), 3)
        roi_val = payload.get("roi_name", "Main_Entrance_ROI")
        is_hit = search_res["is_watchlist_hit"]
        bm = search_res["best_match"]

        if is_hit and bm:
            behavior_summary = (
                f"Şüpheli bekleme ({dwell_val}sn, Verim:{eff_val}) • "
                f"Milvus Eşleşmesi: {bm['name']} (%{bm['cosine_similarity']*100:.1f})"
            )
        else:
            behavior_summary = (
                f"{roi_val} bölgesinde düşük yörünge verimliliği ({eff_val}) • "
                f"İzleme listesinde kayıt bulunamadı"
            )

        event_record = {
            "event_id": f"EVT-{int(time.time() * 1000) % 1000000:06d}",
            "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S"),
            "camera_id": payload.get("camera_id", "CAM-01-GIRIS"),
            "track_id": int(payload.get("track_id", 0)),
            "roi_name": roi_val,
            "dwell_time": dwell_val,
            "efficiency": eff_val,
            "kvkk_compliant": bool(payload.get("kvkk_data_minimized", True)),
            "face_confidence": round(face_conf, 2),
            "demographic_group": demographic_group,
            "embedding_dim": 512,
            "watchlist_hit": is_hit,
            "best_match": bm,
            "top_candidates": search_res["top_candidates"],
            "vector_db_backend": search_res["backend"],
            "search_latency_ms": search_res["search_latency_ms"],
            "behavior_summary": behavior_summary,
            "aligned_face_b64": aligned_b64,
            "cropped_person_b64": crop_b64,
        }

        with self._lock:
            self.events_log.insert(0, event_record)
            if len(self.events_log) > 50:
                self.events_log = self.events_log[:50]

        return event_record


app = FastAPI(
    title="TÜBİTAK 2209-A Hibrit Şüpheli Davranış ve Adil Yüz Tanıma Bulut Servisi",
    version="2.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

cloud_engine = CloudPipelineEngine()


@app.get("/api/v1/health")
def get_health():
    return {
        "status": "ok",
        "project": cloud_engine.config.get("project", {}),
        "milvus": cloud_engine.vector_db.get_status(),
        "edge_telemetry": cloud_engine.edge_telemetry,
        "events_count": len(cloud_engine.events_log),
        "kvkk_privacy_by_design": True,
    }


@app.post("/api/v1/controls")
def update_runtime_controls(update: RuntimeControlUpdate):
    """Allow interactive users in the Web Dashboard to change source, thresholds, or trigger test alerts."""
    if update.source is not None:
        cloud_engine.runtime_controls["requested_source"] = update.source
    if update.time_threshold is not None:
        cloud_engine.runtime_controls["time_threshold"] = float(update.time_threshold)
    if update.efficiency_threshold is not None:
        cloud_engine.runtime_controls["efficiency_threshold"] = float(
            update.efficiency_threshold
        )
    if update.force_trigger_alert:
        cloud_engine.runtime_controls["force_trigger_alert"] = True
    if update.clear_events:
        with cloud_engine._lock:
            cloud_engine.events_log.clear()
    return {
        "status": "updated",
        "controls": cloud_engine.runtime_controls,
        "telemetry": cloud_engine.edge_telemetry,
    }


@app.post("/api/v1/test/image")
async def test_single_image(
    enroll_name: str = Form(""),
    file: UploadFile = File(...),
):
    """Allow any user to upload a photo or webcam capture to test ArcFace 512-D + Milvus matching."""
    content = await file.read()
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Geçersiz görüntü dosyası")

    aligned_face, face_bbox, face_conf = cloud_engine.recognizer.detect_and_align_face(
        img
    )
    emb = cloud_engine.recognizer.extract_embedding(aligned_face, already_aligned=True)
    group = cloud_engine.recognizer.estimate_demographic_group(aligned_face)

    enrolled_record = None
    if enroll_name.strip():
        b64_face = KVKKPrivacyBridge.encode_image_b64(aligned_face)
        enrolled_record = cloud_engine.vector_db.enroll_identity(
            name=enroll_name.strip(),
            embedding=emb,
            risk_level="HIGH",
            demographic_group=group,
            image_b64=b64_face,
            notes="Web arayüzünden anlık test ve kayıt",
        )

    search_res = cloud_engine.vector_db.search_watchlist(
        emb, top_k=3, threshold=cloud_engine.recognizer.match_threshold
    )

    # Also log as a test event so it appears in the Cloud Alerts feed
    event_rec = cloud_engine.process_edge_event(
        {
            "camera_id": "WEB-TEST-UPLOAD",
            "track_id": 999,
            "roi_name": "Interactive_User_Test",
            "dwell_time": 185.0,
            "efficiency": 0.12,
            "kvkk_data_minimized": True,
            "cropped_image_b64": KVKKPrivacyBridge.encode_image_b64(img),
        },
        crop_bgr=img,
    )

    return {
        "face_confidence": round(face_conf, 2),
        "demographic_group": group,
        "embedding_dim": 512,
        "enrolled": enrolled_record,
        "search_result": search_res,
        "aligned_face_b64": event_rec["aligned_face_b64"],
        "event": event_rec,
    }


@app.post("/api/v1/events/analyze")
def analyze_edge_event(payload: EdgeEventPayload):
    event = cloud_engine.process_edge_event(payload.model_dump())
    return event


@app.get("/api/v1/events")
def list_events():
    return {"events": cloud_engine.events_log}


@app.get("/api/v1/watchlist")
def get_watchlist():
    return {
        "status": cloud_engine.vector_db.get_status(),
        "identities": cloud_engine.vector_db.list_identities(),
    }


@app.post("/api/v1/watchlist/enroll")
async def enroll_watchlist(
    name: str = Form(...),
    risk_level: str = Form("HIGH"),
    demographic_group: str = Form("Middle_Eastern"),
    notes: str = Form(""),
    file: UploadFile = File(...),
):
    content = await file.read()
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Geçersiz görüntü dosyası")

    aligned, _, _ = cloud_engine.recognizer.detect_and_align_face(img)
    emb = cloud_engine.recognizer.extract_embedding(aligned, already_aligned=True)
    b64 = KVKKPrivacyBridge.encode_image_b64(aligned)

    res = cloud_engine.vector_db.enroll_identity(
        name=name,
        embedding=emb,
        risk_level=risk_level,
        demographic_group=demographic_group,
        image_b64=b64,
        notes=notes,
    )
    return res


@app.delete("/api/v1/watchlist/{person_id}")
def delete_watchlist_person(person_id: str):
    ok = cloud_engine.vector_db.delete_identity(person_id)
    return {"deleted": ok, "person_id": person_id}


@app.get("/api/v1/metrics/tubitak")
def get_tubitak_metrics():
    if cloud_engine.latest_benchmark is None:
        cloud_engine.latest_benchmark = cloud_engine.evaluator.run_all_benchmarks()
    return cloud_engine.latest_benchmark


@app.post("/api/v1/evaluate/run")
def run_tubitak_evaluation():
    cloud_engine.latest_benchmark = cloud_engine.evaluator.run_all_benchmarks()
    return cloud_engine.latest_benchmark


@app.get("/api/v1/stream/video")
def stream_video():
    """MJPEG live stream of the Edge Analyzer for the Web Dashboard."""

    def generate():
        while True:
            frame_bytes = cloud_engine.latest_jpeg_frame
            if frame_bytes is None:
                canvas = np.zeros((480, 854, 3), dtype=np.uint8)
                canvas[:] = (24, 28, 36)
                cv2.putText(
                    canvas,
                    "TUBITAK 2209-A Uc Katmani (Edge Layer) Akisi Bekleniyor...",
                    (70, 230),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (100, 220, 255),
                    2,
                )
                ok, buf = cv2.imencode(".jpg", canvas)
                if ok:
                    frame_bytes = buf.tobytes()
            if frame_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            time.sleep(0.08)

    return StreamingResponse(
        generate(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>TÜBİTAK 2209-A Cloud Dashboard</h1>")
