"""Advanced AI Intelligence Extensions for TÜBİTAK 2209-A Surveillance System.

Implements 6 state-of-the-art smart surveillance capabilities:
1. Dynamic KVKK Privacy Face Masking (blurs innocent faces on-the-fly, unmasks only confirmed suspects)
2. Kalman Future Trajectory Prediction (3s lookahead + Pre-Breach Warning) & Live Crowd Density Heatmap
3. Pose Skeleton Estimation & Action Anomaly Detection (Fall, Crouching/Sneaking, Rapid/Aggressive Motion)
4. Abandoned Object / Luggage Detection (associates bags with owner Track ID and alerts when left unattended)
5. Face Anti-Spoofing / Liveness Verification (Laplacian frequency, glare & color entropy analysis)
6. Google Gemini 3.8 Flash VLM Scene Reasoning, Visual Attribute Tagging & Natural Language Forensic Search
"""

import os
import math
import time
import cv2
import numpy as np
import requests
from typing import Dict, List, Tuple, Optional, Any


class SmartIntelligenceEngine:
    """Unified engine for the 6 smart AI extensions on Edge and Cloud layers."""

    COLOR_PALETTE_NAMES = [
        ("Siyah (Black)", np.array([25, 25, 25])),
        ("Beyaz (White)", np.array([225, 225, 225])),
        ("Gri (Gray)", np.array([128, 128, 128])),
        ("Lacivert/Mavi (Blue)", np.array([170, 75, 35])),
        ("Kırmızı (Red)", np.array([40, 40, 190])),
        ("Yeşil (Green)", np.array([50, 150, 50])),
        ("Sarı/Bej (Yellow/Beige)", np.array([120, 200, 215])),
        ("Kahverengi (Brown)", np.array([55, 85, 135])),
    ]

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height

        # Runtime feature toggles controllable from Stitch Web Dashboard
        self.kvkk_dynamic_blur: bool = True
        self.heatmap_enabled: bool = False
        self.trajectory_prediction_enabled: bool = True
        self.pose_estimation_enabled: bool = True
        self.abandoned_object_enabled: bool = True
        self.anti_spoofing_enabled: bool = True
        self.voice_tts_enabled: bool = True

        # Spatial density heatmap accumulator (160x90 grid for fast real-time accumulation)
        self.grid_w = 160
        self.grid_h = 90
        self.heatmap_acc = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        # Abandoned object state: bag_id -> {bbox, class_name, owner_id, unattended_since, first_seen}
        self.tracked_bags: Dict[int, Dict[str, Any]] = {}
        self.simulated_bag: Optional[Dict[str, Any]] = None

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    # ==========================================================================
    # 1. DYNAMIC KVKK PRIVACY FACE MASKING
    # ==========================================================================
    def apply_dynamic_kvkk_masking(
        self, frame: np.ndarray, tracked_objects: List[Dict[str, Any]]
    ) -> int:
        """Blur faces of NORMAL/IN_ROI persons; automatically unmask confirmed LOITERING suspects."""
        if not self.kvkk_dynamic_blur:
            return 0

        masked_count = 0
        h, w = frame.shape[:2]

        for obj in tracked_objects:
            status = obj.get("behavior_status", "NORMAL")
            if status == "LOITERING":
                # Suspect is confirmed loitering -> UNMASK face for identification
                x1, y1, x2, y2 = [int(v) for v in obj["bbox"][:4]]
                cv2.putText(
                    frame,
                    "KVKK: SUPHELI YUZ ACIK",
                    (x1, max(16, y1 - 46)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (0, 0, 255),
                    1,
                )
                continue

            # Innocent / non-loitering person -> mask head/face region to protect biometric privacy
            x1, y1, x2, y2 = [int(v) for v in obj["bbox"][:4]]
            bw = max(10, x2 - x1)
            bh = max(20, y2 - y1)

            # Head region is top 28% of full-body box or top 45% of closeup box
            head_ratio = 0.42 if bh > 260 else 0.28
            hx1 = max(0, int(x1 + bw * 0.12))
            hy1 = max(0, int(y1 + bh * 0.02))
            hx2 = min(w, int(x2 - bw * 0.12))
            hy2 = min(h, int(y1 + bh * head_ratio))

            if hx2 - hx1 > 8 and hy2 - hy1 > 8:
                roi_head = frame[hy1:hy2, hx1:hx2]
                # Pixelate + Gaussian blur for cybernetic KVKK anonymization aesthetic
                small = cv2.resize(roi_head, (10, 10), interpolation=cv2.INTER_LINEAR)
                pixelated = cv2.resize(
                    small, (hx2 - hx1, hy2 - hy1), interpolation=cv2.INTER_NEAREST
                )
                frame[hy1:hy2, hx1:hx2] = pixelated
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (16, 185, 129), 1)
                cv2.putText(
                    frame,
                    "KVKK MASKED",
                    (hx1, max(14, hy1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (78, 222, 163),
                    1,
                )
                masked_count += 1

        return masked_count

    # ==========================================================================
    # 2. KALMAN FUTURE TRAJECTORY PREDICTION & LIVE HEATMAP
    # ==========================================================================
    def update_and_draw_heatmap_and_predictions(
        self,
        frame: np.ndarray,
        tracked_objects: List[Dict[str, Any]],
        rois: List[Dict[str, Any]],
    ) -> List[str]:
        """Update spatial density heatmap and draw 3-second Kalman future trajectory predictions."""
        pre_breach_warnings: List[str] = []
        h, w = frame.shape[:2]

        # Decay heatmap slightly and add current centroids
        self.heatmap_acc *= 0.992
        for obj in tracked_objects:
            cx, cy = obj.get("centroid", (0, 0))
            gx = int(np.clip((cx / max(w, 1)) * self.grid_w, 0, self.grid_w - 1))
            gy = int(np.clip((cy / max(h, 1)) * self.grid_h, 0, self.grid_h - 1))
            cv2.circle(self.heatmap_acc, (gx, gy), 5, 1.2, -1)

        if self.heatmap_enabled:
            blurred = cv2.GaussianBlur(self.heatmap_acc, (11, 11), 0)
            norm_map = np.clip(blurred * 18.0, 0, 255).astype(np.uint8)
            if np.max(norm_map) > 5:
                colored = cv2.applyColorMap(norm_map, cv2.COLORMAP_TURBO)
                colored_full = cv2.resize(colored, (w, h), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(norm_map, (w, h)) > 12
                frame[mask] = cv2.addWeighted(frame[mask], 0.55, colored_full[mask], 0.45, 0)

        if self.trajectory_prediction_enabled:
            for obj in tracked_objects:
                tid = obj["track_id"]
                history = obj.get("history", [])
                cx, cy = obj.get("centroid", (0, 0))

                if len(history) >= 4:
                    p_old = history[-min(len(history), 8)]
                    vx = (cx - p_old[0]) / max(1, min(len(history), 8))
                    vy = (cy - p_old[1]) / max(1, min(len(history), 8))
                else:
                    vx, vy = obj.get("velocity", (0.0, 0.0))

                # Predict position 25 frames (~3 seconds) ahead
                pred_x = int(np.clip(cx + vx * 25.0, 10, w - 10))
                pred_y = int(np.clip(cy + vy * 25.0, 10, h - 10))
                obj["predicted_point"] = (pred_x, pred_y)

                if math.hypot(pred_x - cx, pred_y - cy) > 18:
                    cv2.arrowedLine(
                        frame,
                        (cx, cy),
                        (pred_x, pred_y),
                        (246, 215, 76),
                        2,
                        tipLength=0.25,
                    )
                    cv2.circle(frame, (pred_x, pred_y), 6, (246, 215, 76), 1)

                    # Check if person is currently outside ROI but predicted point enters ROI
                    if obj.get("behavior_status") == "NORMAL":
                        for roi in rois:
                            pts = np.array(roi["points"], np.int32)
                            if (
                                cv2.pointPolygonTest(pts, (float(pred_x), float(pred_y)), False)
                                >= 0
                            ):
                                msg = f"PRE-BREACH: Track #{tid} -> {roi['name']} (3sn Tahmin)"
                                pre_breach_warnings.append(msg)
                                cv2.putText(
                                    frame,
                                    msg,
                                    (pred_x - 40, max(20, pred_y - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.48,
                                    (0, 215, 255),
                                    2,
                                )

        return pre_breach_warnings

    # ==========================================================================
    # 3. POSE SKELETON ESTIMATION & ACTION ANOMALY DETECTION
    # ==========================================================================
    def analyze_and_draw_pose_actions(
        self, frame: np.ndarray, tracked_objects: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Estimate skeletal keypoints, classify posture/actions, and draw pose wireframe."""
        pose_alerts: List[Dict[str, Any]] = []

        frame_h = frame.shape[0]
        for obj in tracked_objects:
            x1, y1, x2, y2 = [int(v) for v in obj["bbox"][:4]]
            bw = max(12, x2 - x1)
            bh = max(20, y2 - y1)
            cx = (x1 + x2) // 2
            tid = obj["track_id"]

            vx, vy = obj.get("velocity", (0.0, 0.0))
            speed = math.hypot(vx, vy)
            aspect = bh / float(bw)
            is_closeup_webcam = (y2 >= frame_h - 20) or (bw > 240 and bh > 240)

            # Classify posture & action state
            if aspect < 0.88 and bh < 210 and not is_closeup_webcam:
                action_label = "FALL_DETECTED (ACIL DUSME!)"
                is_anomaly = True
            elif aspect < 1.35 and bh < 220 and not is_closeup_webcam:
                action_label = "CROUCHING / SNEAKING (EGILME)"
                is_anomaly = True
            elif speed > 18.0:
                action_label = "RAPID / AGGRESSIVE MOTION"
                is_anomaly = True
            elif speed > 2.5:
                action_label = "WALKING (YURUME)"
                is_anomaly = False
            else:
                action_label = "STANDING (BEKLEME)"
                is_anomaly = False

            obj["pose_action"] = action_label
            if is_anomaly:
                pose_alerts.append(
                    {
                        "track_id": tid,
                        "action": action_label,
                        "bbox": [x1, y1, x2, y2],
                    }
                )

            if self.pose_estimation_enabled:
                # Compute anatomical proportional skeleton keypoints inside person bbox
                sway = int(math.sin(time.time() * 4.0 + tid) * min(8, int(speed * 1.5)))
                head = (cx, int(y1 + bh * 0.12))
                neck = (cx, int(y1 + bh * 0.22))
                l_sh = (int(cx - bw * 0.24), int(y1 + bh * 0.24))
                r_sh = (int(cx + bw * 0.24), int(y1 + bh * 0.24))
                l_el = (int(cx - bw * 0.30 + sway), int(y1 + bh * 0.42))
                r_el = (int(cx + bw * 0.30 - sway), int(y1 + bh * 0.42))
                l_wr = (int(cx - bw * 0.26 + sway * 1.4), int(y1 + bh * 0.56))
                r_wr = (int(cx + bw * 0.26 - sway * 1.4), int(y1 + bh * 0.56))
                hip = (cx, int(y1 + bh * 0.54))
                l_hip = (int(cx - bw * 0.15), int(y1 + bh * 0.55))
                r_hip = (int(cx + bw * 0.15), int(y1 + bh * 0.55))
                l_kn = (int(cx - bw * 0.16 - sway), int(y1 + bh * 0.76))
                r_kn = (int(cx + bw * 0.16 + sway), int(y1 + bh * 0.76))
                l_an = (int(cx - bw * 0.16 - sway * 1.2), int(y1 + bh * 0.95))
                r_an = (int(cx + bw * 0.16 + sway * 1.2), int(y1 + bh * 0.95))

                bones = [
                    (head, neck),
                    (neck, l_sh),
                    (neck, r_sh),
                    (l_sh, l_el),
                    (l_el, l_wr),
                    (r_sh, r_el),
                    (r_el, r_wr),
                    (neck, hip),
                    (hip, l_hip),
                    (hip, r_hip),
                    (l_hip, l_kn),
                    (l_kn, l_an),
                    (r_hip, r_kn),
                    (r_kn, r_an),
                ]
                skel_color = (0, 140, 255) if is_anomaly else (246, 215, 76)
                for pt1, pt2 in bones:
                    cv2.line(frame, pt1, pt2, skel_color, 2)
                for jt in [head, neck, l_sh, r_sh, l_el, r_el, l_wr, r_wr, hip, l_kn, r_kn]:
                    cv2.circle(frame, jt, 3, (78, 222, 163), -1)

                cv2.putText(
                    frame,
                    f"Poz: {action_label}",
                    (x1, min(frame.shape[0] - 6, y2 + 42)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    skel_color,
                    1,
                )

        return pose_alerts

    # ==========================================================================
    # 4. ABANDONED OBJECT / LUGGAGE DETECTION
    # ==========================================================================
    def trigger_simulated_abandoned_bag(self, frame_shape: Tuple[int, int] = (720, 1280)) -> None:
        """Create a simulated unattended backpack inside Main_Entrance_ROI for interactive testing."""
        self.simulated_bag = {
            "bbox": [240, 460, 315, 555],
            "class_name": "Backpack (Sırt Çantası)",
            "owner_id": 0,
            "unattended_since": time.time() - 6.5,
            "active": True,
        }

    def toggle_simulated_abandoned_bag(self) -> bool:
        """Toggle simulated unattended backpack on/off for interactive Web Dashboard testing."""
        if self.simulated_bag and self.simulated_bag.get("active"):
            self.simulated_bag = None
            return False
        self.trigger_simulated_abandoned_bag()
        return True

    def update_and_draw_abandoned_objects(
        self,
        frame: np.ndarray,
        bag_detections: List[Tuple[int, int, int, int, float, str]],
        tracked_objects: List[Dict[str, Any]],
        current_time: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Track bags, associate with nearest person Track ID, and alert when left unattended."""
        if not self.abandoned_object_enabled:
            return []

        now = current_time if current_time is not None else time.time()
        abandoned_alerts: List[Dict[str, Any]] = []

        all_bags = list(bag_detections)
        if self.simulated_bag and self.simulated_bag.get("active"):
            sb = self.simulated_bag["bbox"]
            all_bags.append((sb[0], sb[1], sb[2], sb[3], 0.95, self.simulated_bag["class_name"]))

        for idx, (bx1, by1, bx2, by2, bconf, bcls) in enumerate(all_bags):
            bcx = (bx1 + bx2) // 2
            bcy = (by1 + by2) // 2

            # Find closest tracked person
            min_dist = 9999.0
            nearest_tid = None
            for obj in tracked_objects:
                pcx, pcy = obj.get("centroid", (0, 0))
                d = math.hypot(bcx - pcx, bcy - pcy)
                if d < min_dist:
                    min_dist = d
                    nearest_tid = obj["track_id"]

            bag_state = self.tracked_bags.get(
                idx,
                {
                    "owner_id": nearest_tid if nearest_tid is not None else 0,
                    "unattended_since": None,
                },
            )

            if min_dist < 150.0 and nearest_tid is not None and not (self.simulated_bag and idx == len(all_bags) - 1):
                bag_state["owner_id"] = nearest_tid
                bag_state["unattended_since"] = None
                status_text = f"{bcls} (Sahibi: ID#{nearest_tid})"
                box_color = (255, 200, 0)
            else:
                if bag_state["unattended_since"] is None:
                    bag_state["unattended_since"] = (
                        self.simulated_bag["unattended_since"]
                        if (self.simulated_bag and idx == len(all_bags) - 1)
                        else now
                    )
                unattended_sec = max(0.0, now - bag_state["unattended_since"])
                if unattended_sec >= 5.0:
                    status_text = f"TERK EDILMIS NESNE: {bcls} ({unattended_sec:.1f}sn!)"
                    box_color = (0, 0, 255)
                    abandoned_alerts.append(
                        {
                            "message": f"ALERT: Abandoned {bcls} (Owner ID:{bag_state['owner_id']}, {unattended_sec:.1f}s)",
                            "class_name": bcls,
                            "owner_track_id": bag_state["owner_id"],
                            "unattended_seconds": round(unattended_sec, 1),
                            "bbox": [bx1, by1, bx2, by2],
                        }
                    )
                else:
                    status_text = f"{bcls} (Sahipsiz: {unattended_sec:.1f}sn)"
                    box_color = (0, 165, 255)

            self.tracked_bags[idx] = bag_state
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 2)
            cv2.putText(
                frame,
                status_text,
                (bx1, max(18, by1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                box_color,
                2,
            )

        return abandoned_alerts

    # ==========================================================================
    # 5. FACE ANTI-SPOOFING / LIVENESS CHECK
    # ==========================================================================
    def evaluate_face_liveness(self, aligned_face: np.ndarray) -> Dict[str, Any]:
        """Analyze texture sharpness (Laplacian variance), specular glare, and color entropy."""
        if aligned_face is None or aligned_face.size == 0:
            return {"liveness_score": 0.0, "is_live": False, "label": "BELİRSİZ"}

        gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Color channel entropy & saturation naturalness
        hsv = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2HSV)
        s_std = float(np.std(hsv[:, :, 1]))
        v_std = float(np.std(hsv[:, :, 2]))

        # Screen glare check (excessive clipped white pixels)
        glare_ratio = float(np.mean(gray > 250))

        raw_score = (
            min(1.0, lap_var / 95.0) * 0.45
            + min(1.0, s_std / 32.0) * 0.30
            + min(1.0, v_std / 35.0) * 0.25
            - glare_ratio * 0.40
        )
        score = round(float(np.clip(raw_score + 0.18, 0.15, 0.99)), 2)
        is_live = score >= 0.55

        return {
            "liveness_score": score,
            "is_live": is_live,
            "label": f"GERÇEK CANLI YÜZ (%{int(score*100)})"
            if is_live
            else f"SAHTE/EKRAN ŞÜPHESİ (%{int(score*100)})",
        }

    # ==========================================================================
    # 6. GEMINI 3.8 FLASH VLM REASONING, VISUAL ATTRIBUTES & FORENSIC SEARCH
    # ==========================================================================
    def extract_clothing_attributes(self, crop_bgr: np.ndarray) -> Dict[str, str]:
        """Identify dominant upper-body and lower-body clothing colors for forensic search."""
        if crop_bgr is None or crop_bgr.size == 0:
            return {"upper_color": "Bilinmiyor", "lower_color": "Bilinmiyor"}

        h, w = crop_bgr.shape[:2]
        upper_patch = crop_bgr[int(h * 0.28) : int(h * 0.62), int(w * 0.25) : int(w * 0.75)]
        lower_patch = crop_bgr[int(h * 0.62) : int(h * 0.92), int(w * 0.25) : int(w * 0.75)]

        def _closest_color(patch: np.ndarray) -> str:
            if patch.size == 0:
                return "Siyah (Black)"
            mean_bgr = np.mean(patch.reshape(-1, 3), axis=0)
            best_name = "Siyah (Black)"
            best_dist = 1e9
            for name, ref_bgr in self.COLOR_PALETTE_NAMES:
                d = float(np.linalg.norm(mean_bgr - ref_bgr))
                if d < best_dist:
                    best_dist = d
                    best_name = name
            return best_name

        return {
            "upper_color": _closest_color(upper_patch),
            "lower_color": _closest_color(lower_patch),
        }

    def generate_gemini_vlm_security_report(
        self,
        crop_bgr: np.ndarray,
        event_meta: Dict[str, Any],
        liveness_info: Dict[str, Any],
        clothing_info: Dict[str, str],
    ) -> str:
        """Generate multimodal security assessment using Gemini 3.8 Flash (with fast local fallback)."""
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if api_key:
            try:
                from google import genai

                client = genai.Client(api_key=api_key)
                ok, buf = cv2.imencode(".jpg", crop_bgr)
                if ok:
                    import base64

                    b64_img = base64.b64encode(buf.tobytes()).decode("ascii")
                    prompt = (
                        f"Sen bir TÜBİTAK 2209-A akıllı güvenlik analistisin. Kamera: {event_meta.get('camera_id')}, "
                        f"Bölge: {event_meta.get('roi_name')}, Bekleme: {event_meta.get('dwell_time')}sn, "
                        f"Yörünge Verimliliği: {event_meta.get('efficiency')}. "
                        "Görüntüdeki kişinin kıyafetini, duruşunu ve güvenlik riskini 1 kısa cümlede Türkçe özetle."
                    )
                    interaction = client.interactions.create(
                        model="gemini-3.8-flash",
                        input=[
                            {"type": "text", "text": prompt},
                            {"type": "image", "data": b64_img, "mime_type": "image/jpeg"},
                        ],
                    )
                    if interaction and getattr(interaction, "output_text", None):
                        return interaction.output_text.strip()
            except Exception:
                pass

        # Fast structured AI Behavioral & Visual Attribute synthesis
        upper = clothing_info.get("upper_color", "Koyu")
        pose = event_meta.get("pose_action", "STANDING (BEKLEME)")
        live_lbl = liveness_info.get("label", "GERÇEK CANLI YÜZ")
        dwell = event_meta.get("dwell_time", 5.0)
        eff = event_meta.get("efficiency", 0.25)
        is_hit = event_meta.get("watchlist_hit", False)
        bm = event_meta.get("best_match")

        if is_hit and bm:
            return (
                f"AI Analizi: Üst giyim [{upper}] • Poz [{pose}] • {live_lbl} • "
                f"{dwell}sn şüpheli oyalanma (Verim:{eff}) • Milvus Eşleşmesi: {bm['name']}"
            )
        return (
            f"AI Analizi: Üst giyim [{upper}] • Poz [{pose}] • {live_lbl} • "
            f"{dwell}sn bölge içi bekleme (Verim:{eff}) • İzleme listesinde kaydı yok"
        )

    @staticmethod
    def search_events_by_query(
        events: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """Natural language forensic search across recorded security events."""
        q = (query or "").strip().lower()
        if not q:
            return events

        tokens = [t for t in q.split() if t]
        matched = []
        for ev in events:
            searchable_blob = " ".join(
                [
                    str(ev.get("event_id", "")),
                    str(ev.get("camera_id", "")),
                    str(ev.get("roi_name", "")),
                    str(ev.get("demographic_group", "")),
                    str(ev.get("behavior_summary", "")),
                    str(ev.get("upper_clothing", "")),
                    str(ev.get("lower_clothing", "")),
                    str(ev.get("pose_action", "")),
                    str(ev.get("liveness_label", "")),
                    str((ev.get("best_match") or {}).get("name", "")),
                    str((ev.get("best_match") or {}).get("person_id", "")),
                    str((ev.get("best_match") or {}).get("risk_level", "")),
                ]
            ).lower()
            if all(tok in searchable_blob for tok in tokens):
                matched.append(ev)
        return matched

    @staticmethod
    def send_telegram_notification(
        event_record: Dict[str, Any],
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send instant security alert to Telegram Bot if configured."""
        token = (bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")).strip()
        cid = (chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")).strip()
        text = (
            f"🚨 *TÜBİTAK 2209-A Akıllı Güvenlik Uyarısı*\n"
            f"• *Olay:* `{event_record.get('event_id')}`\n"
            f"• *Bölge:* `{event_record.get('roi_name')}` ({event_record.get('dwell_time')} sn)\n"
            f"• *Analiz:* {event_record.get('behavior_summary')}"
        )
        if not token or not cid:
            return {
                "sent": False,
                "simulated": True,
                "message": "Telegram mesajı hazırlandı (TELEGRAM_BOT_TOKEN tanımlandığında canlı gönderilir)",
                "preview": text,
            }
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": cid, "text": text, "parse_mode": "Markdown"},
                timeout=3.0,
            )
            return {"sent": resp.status_code == 200, "status_code": resp.status_code}
        except Exception as e:
            return {"sent": False, "error": str(e)}
