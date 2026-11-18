"""Çok Seviyeli Güvenlik Alarmı ve İhbar Yönetim Sistemi (TÜBİTAK 2209-A).

Bu modül uç ve bulut katmanlarından gelen risk skorlarını, oyalanma olaylarını,
Milvus izleme listesi eşleşmelerini, poz anomalilerini ve terk edilmiş nesne ihlallerini
izleyerek çok seviyeli alarmlar üretir, yaşam döngüsünü (AKTİF -> ONAYLANDI -> ÇÖZÜLDÜ)
yönetir ve kalıcı alarm günlüğüne (data/alarms_log.json) kaydeder.
"""

import os
import json
import time
import uuid
import threading
import requests
from typing import Dict, List, Optional, Any, Tuple


class AlarmSystemManager:
    """Gerçek zamanlı alarm tetikleme, önceliklendirme, kalıcı kayıt ve bildirim yöneticisi."""

    LEVEL_PRIORITY = {
        "KRİTİK": 1,
        "YÜKSEK": 2,
        "ORTA": 3,
        "DÜŞÜK": 4,
    }

    def __init__(
        self,
        log_path: str = "data/alarms_log.json",
        cooldown_seconds: float = 15.0,
    ):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.isabs(log_path):
            log_path = os.path.join(base_dir, log_path)
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        self.cooldown_seconds = float(cooldown_seconds)
        self.siren_muted: bool = False
        self.alarms: List[Dict[str, Any]] = []
        self._last_trigger_map: Dict[str, Tuple[float, int]] = {}
        self._lock = threading.Lock()
        self._load_alarms()

    def _load_alarms(self) -> None:
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.alarms = data[:100]
            except Exception as e:
                print(f"[AlarmSistemi] Alarm günlüğü okuma uyarısı: {e}")

    def _save_alarms(self) -> None:
        try:
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(self.alarms[:100], f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[AlarmSistemi] Alarm günlüğü kaydetme uyarısı: {e}")

    @staticmethod
    def determine_alarm_level(risk_percent: float, category: str = "") -> str:
        """Risk yüzdesi ve olay türüne göre alarm seviyesini belirler."""
        cat_u = str(category).upper()
        if (
            risk_percent >= 80.0
            or "KRİTİK" in cat_u
            or "DÜŞME" in cat_u
            or "TERK EDİLMİŞ" in cat_u
            or "MİLVUS" in cat_u
        ):
            return "KRİTİK"
        if risk_percent >= 55.0 or "OYALANMA" in cat_u or "SAHTE" in cat_u:
            return "YÜKSEK"
        if risk_percent >= 30.0 or "ERKEN" in cat_u:
            return "ORTA"
        return "DÜŞÜK"

    def evaluate_and_trigger(
        self,
        title: str,
        description: str,
        category: str,
        risk_percent: float,
        camera_id: str = "CAM-01-GİRİŞ",
        track_id: int = 0,
        roi_name: str = "Ana_Giriş_ROI",
        snapshot_b64: Optional[str] = None,
        matched_name: Optional[str] = None,
        risk_breakdown: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Koşullara göre yeni alarm kaydı oluşturur, kalıcı olarak saklar ve harici kanallara iletir."""
        level = self.determine_alarm_level(risk_percent, category)
        if level == "DÜŞÜK" and not force:
            return None

        now = time.time()
        dedup_key = f"{camera_id}:{track_id}:{category}"
        prio = self.LEVEL_PRIORITY.get(level, 3)

        with self._lock:
            if not force and dedup_key in self._last_trigger_map:
                last_ts, last_prio = self._last_trigger_map[dedup_key]
                # Soğuma süresi içindeyse yalnızca seviye yükseldiyse yeniden alarm üret
                if (now - last_ts) < self.cooldown_seconds and prio >= last_prio:
                    return None

            self._last_trigger_map[dedup_key] = (now, prio)

            alarm_record = {
                "alarm_id": f"ALM-{uuid.uuid4().hex[:6].upper()}",
                "timestamp": now,
                "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S"),
                "seviye": level,
                "oncelik": prio,
                "durum": "AKTİF",
                "kategori": category,
                "baslik": title,
                "aciklama": description,
                "risk_yuzdesi": round(float(risk_percent), 1),
                "risk_kirilimi": risk_breakdown or {},
                "kamera_id": camera_id,
                "iz_id": int(track_id),
                "bolge_adi": roi_name,
                "eslesen_kisi": matched_name,
                "goruntu_b64": snapshot_b64,
                "onay_zamani": None,
                "cozum_zamani": None,
            }

            self.alarms.insert(0, alarm_record)
            if len(self.alarms) > 100:
                self.alarms = self.alarms[:100]
            self._save_alarms()

        # İsteğe bağlı harici Webhook veya Telegram bildirimi
        self._dispatch_external_webhook(alarm_record)
        return alarm_record

    def _dispatch_external_webhook(self, alarm: Dict[str, Any]) -> None:
        webhook_url = os.environ.get("ALARM_WEBHOOK_URL", "").strip()
        if not webhook_url:
            return
        try:
            payload = {k: v for k, v in alarm.items() if k != "goruntu_b64"}
            requests.post(webhook_url, json=payload, timeout=2.0)
        except Exception:
            pass

    def acknowledge_alarm(self, alarm_id: str) -> Optional[Dict[str, Any]]:
        """Operatör tarafından alarmı onaylandı olarak işaretler."""
        with self._lock:
            for alm in self.alarms:
                if alm["alarm_id"] == alarm_id:
                    alm["durum"] = "ONAYLANDI"
                    alm["onay_zamani"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    self._save_alarms()
                    return alm
        return None

    def resolve_alarm(self, alarm_id: str) -> Optional[Dict[str, Any]]:
        """Operatör tarafından alarmı çözüldü/kapatıldı olarak işaretler."""
        with self._lock:
            for alm in self.alarms:
                if alm["alarm_id"] == alarm_id:
                    alm["durum"] = "ÇÖZÜLDÜ"
                    alm["cozum_zamani"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    self._save_alarms()
                    return alm
        return None

    def resolve_all_active(self) -> int:
        """Tüm aktif alarmları topluca çözüldü durumuna geçirir."""
        count = 0
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            for alm in self.alarms:
                if alm["durum"] in ("AKTİF", "ONAYLANDI"):
                    alm["durum"] = "ÇÖZÜLDÜ"
                    alm["cozum_zamani"] = now_str
                    count += 1
            if count > 0:
                self._save_alarms()
        return count

    def clear_history(self) -> None:
        with self._lock:
            self.alarms.clear()
            self._last_trigger_map.clear()
            self._save_alarms()

    def get_summary(self) -> Dict[str, Any]:
        """Aktif, kritik ve toplam alarm istatistiklerini döndürür."""
        with self._lock:
            active = [a for a in self.alarms if a["durum"] == "AKTİF"]
            critical_active = [a for a in active if a["seviye"] == "KRİTİK"]
            high_active = [a for a in active if a["seviye"] == "YÜKSEK"]
            return {
                "toplam_alarm": len(self.alarms),
                "aktif_alarm_sayisi": len(active),
                "kritik_aktif_sayisi": len(critical_active),
                "yuksek_aktif_sayisi": len(high_active),
                "siren_durumu": not self.siren_muted and len(active) > 0,
                "siren_susturuldu": self.siren_muted,
                "en_yuksek_aktif_alarm": active[0] if active else None,
            }

    def list_alarms(self, limit: int = 30) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.alarms[:limit])
