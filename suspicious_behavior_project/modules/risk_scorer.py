"""Kişi Risk Yüzdesi Hesaplama Modülü (TÜBİTAK 2209-A — Uç ve Bulut Katmanı).

Her takip edilen kişi için çok faktörlü ağırlıklı risk puanlama modeli uygular (%0 - %100):
1. Bölge İhlali ve Kalış Süresi Skoru (Maks. %25)
2. Yörünge Verimsizliği / Amaçsız Dolaşma Skoru (Maks. %20)
3. Vücut Pozu ve Hareket Anomalisi Skoru (Maks. %15)
4. Milvus İzleme Listesi Biyometrik Eşleşme Skoru (Maks. %30)
5. Yüz Canlılık / Sahtecilik (Anti-Spoofing) Skoru (Maks. %10)
"""

from typing import Dict, Any, Optional, Tuple


class PersonRiskScorer:
    """Takip edilen kişiler için anlık ve olay bazlı %0 - %100 risk yüzdesi hesaplar."""

    TIER_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
        "DÜŞÜK": (128, 222, 74),     # Yeşil
        "ORTA": (36, 191, 251),      # Sarı-Turuncu
        "YÜKSEK": (0, 120, 255),     # Turuncu-Kırmızı
        "KRİTİK": (68, 68, 239),     # Kırmızı
    }

    @classmethod
    def classify_tier(cls, risk_percent: float) -> str:
        """Risk yüzdesini 4 standart güvenlik seviyesine ayırır."""
        if risk_percent >= 80.0:
            return "KRİTİK"
        if risk_percent >= 55.0:
            return "YÜKSEK"
        if risk_percent >= 30.0:
            return "ORTA"
        return "DÜŞÜK"

    def compute_person_risk(
        self,
        behavior_status: str = "NORMAL",
        roi_name: Optional[str] = None,
        dwell_time: float = 0.0,
        time_threshold: float = 5.0,
        efficiency: float = 1.0,
        pose_action: str = "BEKLEME",
        watchlist_hit: bool = False,
        watchlist_risk_level: Optional[str] = None,
        liveness_is_live: bool = True,
        pre_breach: bool = False,
        abandoned_bag_owner: bool = False,
    ) -> Dict[str, Any]:
        """Çok faktörlü ağırlıklı risk yüzdesini (%0 - %100) ve alt bileşenlerini hesaplar."""
        # 1. Bölge ihlali ve kalış süresi skoru (Maks. %25)
        roi_score = 0.0
        if roi_name or behavior_status in ("BÖLGEDE", "ŞÜPHELİ OYALANMA"):
            ratio = min(1.0, max(0.0, float(dwell_time) / max(float(time_threshold), 0.5)))
            base_roi = 5.0 + ratio * 16.0
            if roi_name and ("Sensitive" in roi_name or "Hassas" in roi_name or "Yasak" in roi_name):
                base_roi += 4.0
            roi_score = min(25.0, base_roi)
        elif pre_breach:
            roi_score = 10.0

        # 2. Yörünge verimsizliği skoru (Maks. %20)
        traj_score = 0.0
        eff = min(1.0, max(0.0, float(efficiency)))
        if roi_name or behavior_status in ("BÖLGEDE", "ŞÜPHELİ OYALANMA"):
            if eff < 0.30:
                traj_score = 14.0 + (0.30 - eff) * 20.0
            else:
                traj_score = (1.0 - eff) * 10.0
        elif eff < 0.25:
            traj_score = 5.0
        traj_score = min(20.0, max(0.0, traj_score))

        # 3. Vücut pozu, hareket anomalisi ve terk edilmiş nesne skoru (Maks. %15)
        pose_score = 0.0
        p_upper = str(pose_action).upper()
        if abandoned_bag_owner or "TERK EDİLMİŞ" in p_upper:
            pose_score = 15.0
        elif "DÜŞME" in p_upper or "EĞİLME" in p_upper or "SIZMA" in p_upper:
            pose_score = 15.0
        elif "AGRESİF" in p_upper or "HIZLI" in p_upper:
            pose_score = 12.0
        elif "BEKLEME" in p_upper and (roi_name or behavior_status != "NORMAL"):
            pose_score = 6.0
        elif "YÜRÜME" in p_upper:
            pose_score = 2.0

        # 4. Milvus izleme listesi biyometrik eşleşme skoru (Maks. %30)
        biometric_score = 0.0
        if watchlist_hit:
            lvl = str(watchlist_risk_level or "HIGH").upper()
            if lvl in ("CRITICAL", "KRİTİK"):
                biometric_score = 30.0
            elif lvl in ("HIGH", "YÜKSEK"):
                biometric_score = 24.0
            else:
                biometric_score = 16.0

        # 5. Yüz canlılık / sahtecilik skoru (Maks. %10)
        spoof_score = 0.0 if liveness_is_live else 10.0

        total_risk = round(
            min(100.0, max(0.0, roi_score + traj_score + pose_score + biometric_score + spoof_score)),
            1,
        )
        tier = self.classify_tier(total_risk)

        return {
            "risk_percent": total_risk,
            "risk_tier": tier,
            "color_bgr": self.TIER_COLORS_BGR[tier],
            "breakdown": {
                "roi_kalma_puani": round(roi_score, 1),
                "yorunge_verimsizlik_puani": round(traj_score, 1),
                "poz_eylem_puani": round(pose_score, 1),
                "biyometrik_milvus_puani": round(biometric_score, 1),
                "sahtecilik_puani": round(spoof_score, 1),
            },
        }
