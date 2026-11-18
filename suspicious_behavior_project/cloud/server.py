"""FastAPI Bulut Katmanı Servisi, Risk Puanlama, Alarm Merkezi ve Web Paneli (TÜBİTAK 2209-A).

Sağladığı servisler:
- KVKK uyumlu Bulut uç noktası (/api/v1/events/analyze): Uç katmanından yalnızca
  kırpılmış şüpheli görüntüsü ve davranış meta verilerini alır.
- FairFace demografik kalibrasyonlu 512-D ArcFace yüz gömme çıkarımı.
- Milvus Vektör Veritabanı (COSINE) izleme listesi araması.
- Çok faktörlü Kişi Risk Yüzdesi (%0-%100) hesaplaması ve çok seviyeli Alarm Yönetimi.
- Canlı MJPEG video akışı (/api/v1/stream/video) ve Google Stitch Web Gösterge Paneli (/).
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
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJE_KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJE_KOK not in sys.path:
    sys.path.insert(0, PROJE_KOK)

from utils.helpers import load_config
from utils.demo_video_generator import ensure_demo_assets
from modules.face_recognizer import FairFaceRecognizer
from modules.vector_db import MilvusWatchlistDB
from modules.privacy_bridge import KVKKPrivacyBridge
from modules.fairness_evaluator import TubitakEvaluationSuite
from modules.smart_analytics import SmartIntelligenceEngine
from modules.risk_scorer import PersonRiskScorer
from modules.alarm_manager import AlarmSystemManager


class EdgeEventPayload(BaseModel):
    camera_id: str = "KAM-01-GİRİŞ"
    track_id: int
    roi_name: str
    dwell_time: float
    efficiency: float
    timestamp: Optional[float] = None
    kvkk_data_minimized: bool = True
    raw_stream_transmitted: bool = False
    cropped_image_b64: str
    pose_action: Optional[str] = "BEKLEME"
    edge_risk_percent: Optional[float] = 0.0


class RuntimeControlUpdate(BaseModel):
    source: Optional[str] = None
    time_threshold: Optional[float] = None
    efficiency_threshold: Optional[float] = None
    force_trigger_alert: Optional[bool] = None
    clear_events: Optional[bool] = None
    kvkk_dynamic_blur: Optional[bool] = None
    heatmap_enabled: Optional[bool] = None
    trajectory_prediction_enabled: Optional[bool] = None
    pose_estimation_enabled: Optional[bool] = None
    simulate_abandoned_bag: Optional[bool] = None


class CloudPipelineEngine:
    """ArcFace, Milvus İzleme Listesi, Risk Motoru ve Alarm Yöneticisini koordine eden Bulut Katmanı."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        fr_cfg = self.config.get("face_recognition", {})
        mv_cfg = self.config.get("milvus", {})
        alm_cfg = self.config.get("alarm_system", {})

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
        self.smart_engine = SmartIntelligenceEngine()
        self.risk_scorer = PersonRiskScorer()
        self.alarm_manager = AlarmSystemManager(
            log_path=alm_cfg.get("log_path", "data/alarms_log.json"),
            cooldown_seconds=float(alm_cfg.get("cooldown_seconds", 15.0)),
        )

        self.events_log: List[Dict[str, Any]] = []
        self.active_person_risks: List[Dict[str, Any]] = []
        self.latest_benchmark: Optional[Dict[str, Any]] = None
        self.edge_telemetry: Dict[str, Any] = {
            "fps": 0.0,
            "active_detections": 0,
            "active_tracks": 0,
            "frames_processed_on_edge": 0,
            "max_active_risk_percent": 0.0,
            "active_source": "auto",
            "time_threshold": 5.0,
            "efficiency_threshold": 0.30,
            "camera_id": self.config["camera"].get("camera_id", "KAM-01-GİRİŞ"),
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
        """Örnek senaryonun anında çalışması için varsayılan izleme listesi kaydını oluşturur."""
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
                    risk_level="KRİTİK",
                    demographic_group="Orta_Dogulu",
                    image_b64=b64,
                    notes="Kritik tesis giriş bölgesi (Ana_Giris_Bolgesi) aranan şahıs profili",
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
        """Uç katmanından gelen KVKK uyumlu şüpheli olayını işler, risk yüzdesini hesaplar ve alarm üretir."""
        if crop_bgr is None:
            crop_bgr = self.decode_b64_image(payload.get("cropped_image_b64", ""))

        aligned_face, _face_bbox, face_conf = self.recognizer.detect_and_align_face(
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
        roi_val = payload.get("roi_name", "Ana_Giris_Bolgesi")
        pose_val = payload.get("pose_action", "BEKLEME")
        cam_id = payload.get("camera_id", "KAM-01-GİRİŞ")
        track_id = int(payload.get("track_id", 0))
        is_hit = search_res["is_watchlist_hit"]
        bm = search_res["best_match"]

        liveness_info = self.smart_engine.evaluate_face_liveness(aligned_face)
        clothing_info = self.smart_engine.extract_clothing_attributes(crop_bgr)

        # Çok faktörlü Kişi Risk Yüzdesi (%0 - %100) hesabı
        risk_info = self.risk_scorer.compute_person_risk(
            behavior_status="ŞÜPHELİ OYALANMA",
            roi_name=roi_val,
            dwell_time=dwell_val,
            time_threshold=float(self.edge_telemetry.get("time_threshold", 5.0)),
            efficiency=eff_val,
            pose_action=pose_val,
            watchlist_hit=is_hit,
            watchlist_risk_level=(bm or {}).get("risk_level"),
            liveness_is_live=liveness_info["is_live"],
            abandoned_bag_owner=("Terk_Edilmis" in str(roi_val) or "TERK EDİLMİŞ" in str(pose_val)),
        )

        meta_for_vlm = {
            "camera_id": cam_id,
            "roi_name": roi_val,
            "dwell_time": dwell_val,
            "efficiency": eff_val,
            "pose_action": pose_val,
            "risk_percent": risk_info["risk_percent"],
            "risk_tier": risk_info["risk_tier"],
            "watchlist_hit": is_hit,
            "best_match": bm,
        }
        behavior_summary = self.smart_engine.generate_gemini_vlm_security_report(
            crop_bgr, meta_for_vlm, liveness_info, clothing_info
        )

        event_record = {
            "event_id": f"OLY-{int(time.time() * 1000) % 1000000:06d}",
            "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S"),
            "camera_id": cam_id,
            "track_id": track_id,
            "roi_name": roi_val,
            "dwell_time": dwell_val,
            "efficiency": eff_val,
            "kvkk_compliant": bool(payload.get("kvkk_data_minimized", True)),
            "face_confidence": round(face_conf, 2),
            "demographic_group": demographic_group,
            "embedding_dim": 512,
            "liveness_score": liveness_info["liveness_score"],
            "liveness_is_live": liveness_info["is_live"],
            "liveness_label": liveness_info["label"],
            "upper_clothing": clothing_info["upper_color"],
            "lower_clothing": clothing_info["lower_color"],
            "pose_action": pose_val,
            "risk_percent": risk_info["risk_percent"],
            "risk_tier": risk_info["risk_tier"],
            "risk_breakdown": risk_info["breakdown"],
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

        # Çok seviyeli Alarm ve İhbar Sistemini tetikle
        if is_hit and bm:
            alarm_title = f"Milvus İzleme Listesi Eşleşmesi: {bm['name']}"
            alarm_cat = f"MİLVUS EŞLEŞMESİ ({bm.get('risk_level', 'YÜKSEK')})"
        elif "Terk_Edilmis" in str(roi_val) or "TERK EDİLMİŞ" in str(pose_val):
            alarm_title = f"Terk Edilmiş Şüpheli Nesne (İz #{track_id})"
            alarm_cat = "TERK EDİLMİŞ NESNE"
        else:
            alarm_title = f"{roi_val} Bölgesinde Şüpheli Oyalanma (İz #{track_id})"
            alarm_cat = "ŞÜPHELİ OYALANMA"

        self.alarm_manager.evaluate_and_trigger(
            title=alarm_title,
            description=behavior_summary,
            category=alarm_cat,
            risk_percent=risk_info["risk_percent"],
            camera_id=cam_id,
            track_id=track_id,
            roi_name=roi_val,
            snapshot_b64=aligned_b64,
            matched_name=(bm or {}).get("name"),
            risk_breakdown=risk_info["breakdown"],
        )

        return event_record


app = FastAPI(
    title="TÜBİTAK 2209-A Hibrit Şüpheli Davranış, Risk Skorlama ve Adil Yüz Tanıma Bulut Servisi",
    version="3.0.0",
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
    se = cloud_engine.smart_engine
    return {
        "status": "ok",
        "project": cloud_engine.config.get("project", {}),
        "milvus": cloud_engine.vector_db.get_status(),
        "edge_telemetry": cloud_engine.edge_telemetry,
        "active_person_risks": cloud_engine.active_person_risks,
        "alarm_summary": cloud_engine.alarm_manager.get_summary(),
        "events_count": len(cloud_engine.events_log),
        "kvkk_privacy_by_design": True,
        "smart_features": {
            "kvkk_dynamic_blur": se.kvkk_dynamic_blur,
            "heatmap_enabled": se.heatmap_enabled,
            "trajectory_prediction_enabled": se.trajectory_prediction_enabled,
            "pose_estimation_enabled": se.pose_estimation_enabled,
            "abandoned_object_enabled": se.abandoned_object_enabled,
            "anti_spoofing_enabled": se.anti_spoofing_enabled,
            "simulated_abandoned_bag": bool(se.simulated_bag and se.simulated_bag.get("active")),
        },
    }


@app.post("/api/v1/controls")
def update_runtime_controls(update: RuntimeControlUpdate):
    """Web panelinden video kaynağını, bekleme eşiğini ve akıllı YZ modüllerini anlık yönetir."""
    se = cloud_engine.smart_engine
    if update.source is not None:
        cloud_engine.runtime_controls["requested_source"] = update.source
    if update.time_threshold is not None:
        cloud_engine.runtime_controls["time_threshold"] = float(update.time_threshold)
        cloud_engine.edge_telemetry["time_threshold"] = float(update.time_threshold)
    if update.efficiency_threshold is not None:
        cloud_engine.runtime_controls["efficiency_threshold"] = float(
            update.efficiency_threshold
        )
    if update.force_trigger_alert:
        cloud_engine.runtime_controls["force_trigger_alert"] = True
    if update.clear_events:
        with cloud_engine._lock:
            cloud_engine.events_log.clear()
        se.simulated_bag = None
    if update.kvkk_dynamic_blur is not None:
        se.kvkk_dynamic_blur = bool(update.kvkk_dynamic_blur)
    if update.heatmap_enabled is not None:
        se.heatmap_enabled = bool(update.heatmap_enabled)
    if update.trajectory_prediction_enabled is not None:
        se.trajectory_prediction_enabled = bool(update.trajectory_prediction_enabled)
    if update.pose_estimation_enabled is not None:
        se.pose_estimation_enabled = bool(update.pose_estimation_enabled)
    if update.simulate_abandoned_bag:
        se.toggle_simulated_abandoned_bag()
    return {
        "status": "güncellendi",
        "controls": cloud_engine.runtime_controls,
        "smart_features": {
            "kvkk_dynamic_blur": se.kvkk_dynamic_blur,
            "heatmap_enabled": se.heatmap_enabled,
            "trajectory_prediction_enabled": se.trajectory_prediction_enabled,
            "pose_estimation_enabled": se.pose_estimation_enabled,
            "simulated_abandoned_bag": bool(se.simulated_bag and se.simulated_bag.get("active")),
        },
        "telemetry": cloud_engine.edge_telemetry,
    }


@app.get("/api/v1/alarms")
def list_alarms():
    """Tüm alarmları ve özet istatistikleri döndürür."""
    return {
        "summary": cloud_engine.alarm_manager.get_summary(),
        "alarms": cloud_engine.alarm_manager.list_alarms(limit=40),
    }


@app.post("/api/v1/alarms/test")
def trigger_test_alarm():
    """Alarm ve siren sistemini anında doğrulamak için test alarmı oluşturur."""
    alm = cloud_engine.alarm_manager.evaluate_and_trigger(
        title="Acil Güvenlik Alarmı Testi (Operatör Tetiklemesi)",
        description="Çok seviyeli alarm, siren ve ihbar hattı operatör tarafından test edildi. Risk: %88 [KRİTİK].",
        category="KRİTİK TEST ALARMI",
        risk_percent=88.0,
        camera_id=cloud_engine.edge_telemetry.get("camera_id", "KAM-01-GİRİŞ"),
        track_id=0,
        roi_name="Hassas_Bolge",
        risk_breakdown={
            "roi_kalma_puani": 22.0,
            "yorunge_verimsizlik_puani": 18.0,
            "poz_eylem_puani": 18.0,
            "biyometrik_milvus_puani": 30.0,
            "sahtecilik_puani": 0.0,
        },
        force=True,
    )
    return {"alarm": alm, "summary": cloud_engine.alarm_manager.get_summary()}


@app.post("/api/v1/alarms/{alarm_id}/acknowledge")
def acknowledge_alarm(alarm_id: str):
    alm = cloud_engine.alarm_manager.acknowledge_alarm(alarm_id)
    if not alm:
        raise HTTPException(status_code=404, detail="Alarm bulunamadı")
    return {"alarm": alm, "summary": cloud_engine.alarm_manager.get_summary()}


@app.post("/api/v1/alarms/{alarm_id}/resolve")
def resolve_alarm(alarm_id: str):
    alm = cloud_engine.alarm_manager.resolve_alarm(alarm_id)
    if not alm:
        raise HTTPException(status_code=404, detail="Alarm bulunamadı")
    return {"alarm": alm, "summary": cloud_engine.alarm_manager.get_summary()}


@app.post("/api/v1/alarms/resolve-all")
def resolve_all_alarms():
    resolved_count = cloud_engine.alarm_manager.resolve_all_active()
    return {
        "resolved_count": resolved_count,
        "summary": cloud_engine.alarm_manager.get_summary(),
    }


@app.post("/api/v1/alarms/clear")
def clear_alarm_history():
    cloud_engine.alarm_manager.clear_history()
    return {"status": "temizlendi", "summary": cloud_engine.alarm_manager.get_summary()}


@app.post("/api/v1/alarms/mute")
def toggle_alarm_mute():
    cloud_engine.alarm_manager.siren_muted = not cloud_engine.alarm_manager.siren_muted
    return {
        "siren_susturuldu": cloud_engine.alarm_manager.siren_muted,
        "summary": cloud_engine.alarm_manager.get_summary(),
    }


@app.get("/api/v1/alarms/export")
def export_alarms_json():
    return JSONResponse(
        content={
            "proje": "TÜBİTAK 2209-A Akıllı Güvenlik ve Alarm Raporu",
            "olusturma_zamani": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ozet": cloud_engine.alarm_manager.get_summary(),
            "alarmlar": [
                {k: v for k, v in a.items() if k != "goruntu_b64"}
                for a in cloud_engine.alarm_manager.list_alarms(limit=100)
            ],
        },
        headers={"Content-Disposition": "attachment; filename=tubitak_alarm_raporu.json"},
    )


@app.get("/api/v1/events/search")
def search_events(q: str = ""):
    """Kayıtlı güvenlik olayları üzerinde doğal dilde adli arama yapar."""
    filtered = cloud_engine.smart_engine.search_events_by_query(
        cloud_engine.events_log, q
    )
    return {"query": q, "count": len(filtered), "events": filtered}


@app.post("/api/v1/notify/telegram")
def notify_telegram():
    """En güncel güvenlik uyarısını Telegram botuna gönderir."""
    if not cloud_engine.events_log:
        return {"sent": False, "message": "Henüz gönderilecek bir uyarı kaydı yok."}
    return cloud_engine.smart_engine.send_telegram_notification(
        cloud_engine.events_log[0]
    )


@app.post("/api/v1/test/image")
async def test_single_image(
    enroll_name: str = Form(""),
    file: UploadFile = File(...),
):
    """Yüklenen tekil fotoğraf üzerinde ArcFace 512-D, canlılık, risk puanı ve Milvus araması yapar."""
    content = await file.read()
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Geçersiz görüntü dosyası")

    aligned_face, _face_bbox, face_conf = cloud_engine.recognizer.detect_and_align_face(
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
            risk_level="YÜKSEK",
            demographic_group=group,
            image_b64=b64_face,
            notes="Web arayüzünden anlık test ve kayıt",
        )

    search_res = cloud_engine.vector_db.search_watchlist(
        emb, top_k=3, threshold=cloud_engine.recognizer.match_threshold
    )

    event_rec = cloud_engine.process_edge_event(
        {
            "camera_id": "WEB-TEST-YÜKLEME",
            "track_id": 999,
            "roi_name": "Etkilesimli_Kullanici_Testi",
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
    return cloud_engine.process_edge_event(payload.model_dump())


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
    risk_level: str = Form("YÜKSEK"),
    demographic_group: str = Form("Orta_Dogulu"),
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

    return cloud_engine.vector_db.enroll_identity(
        name=name,
        embedding=emb,
        risk_level=risk_level,
        demographic_group=demographic_group,
        image_b64=b64,
        notes=notes,
    )


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
    """Web Gösterge Paneli için Uç Katmanı canlı MJPEG video akışı."""

    def generate():
        while True:
            frame_bytes = cloud_engine.latest_jpeg_frame
            if frame_bytes is None:
                canvas = np.zeros((480, 854, 3), dtype=np.uint8)
                canvas[:] = (24, 28, 36)
                cv2.putText(
                    canvas,
                    "TUBITAK 2209-A Uc Katmani Akisi Bekleniyor...",
                    (95, 230),
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
    return HTMLResponse("<h1>TÜBİTAK 2209-A Akıllı Güvenlik Paneli</h1>")
