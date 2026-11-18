"""DeepSORT Çoklu Nesne Takip Modülü — Kalman Filtresi ve Re-ID Görünüm Eşleştirme.

TÜBİTAK 2209-A Uç Katmanı Kişi Takibi gereksinimlerini uygular:
- 8 durumlu Kalman Filtresi kestirimi (cx, cy, w, h, vx, vy, vw, vh)
- Kısa süreli engellemelerde (occlusion) kimlik korumak için 512 boyutlu L2-normalize
  Re-ID görünüm öznitelik çıkarımı
- IoU, hareket eşiği ve kosinüs görünüm mesafesini birleştiren Macar (Hungarian) ataması
- Amaçsız bekleme ve yörünge verimliliği analizi için tam hareket geçmişi kaydı
"""

import cv2
import numpy as np
from scipy.spatial import distance as dist
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Optional, Any


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """İki [xmin, ymin, xmax, ymax] kutu kümesi arasında ikili IoU matrisini hesaplar."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = np.maximum(1.0, (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1]))
    area_b = np.maximum(1.0, (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1]))

    union_area = area_a[:, None] + area_b[None, :] - inter_area
    return inter_area / np.maximum(union_area, 1e-6)


class ReIDAppearanceExtractor:
    """Kişi kırpımından 512 boyutlu L2-normalize Re-ID görünüm vektörü çıkarır."""

    def __init__(self, feature_dim: int = 512):
        self.feature_dim = feature_dim
        rng = np.random.RandomState(42)
        self._proj = rng.randn(384, feature_dim).astype(np.float32) / np.sqrt(384)

    def extract(self, frame: Optional[np.ndarray], bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Kişi sınırlayıcı kutusu için 512-D birim görünüm öznitelik vektörü döndürür."""
        if frame is None or frame.size == 0:
            vec = np.ones(self.feature_dim, dtype=np.float32)
            return vec / np.linalg.norm(vec)

        h, w = frame.shape[:2]
        x1 = max(0, min(int(bbox[0]), w - 2))
        y1 = max(0, min(int(bbox[1]), h - 2))
        x2 = max(x1 + 1, min(int(bbox[2]), w))
        y2 = max(y1 + 1, min(int(bbox[3]), h))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            vec = np.ones(self.feature_dim, dtype=np.float32)
            return vec / np.linalg.norm(vec)

        resized = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Üst (baş/omuz), orta (gövde) ve alt (bacaklar) yatay şeritleri
        stripes = [hsv[0:42, :], hsv[42:85, :], hsv[85:128, :]]
        feats = []
        for stripe in stripes:
            h_hist = cv2.calcHist([stripe], [0], None, [32], [0, 180]).flatten()
            s_hist = cv2.calcHist([stripe], [1], None, [32], [0, 256]).flatten()
            v_hist = cv2.calcHist([stripe], [2], None, [32], [0, 256]).flatten()
            stripe_feat = np.concatenate([h_hist, s_hist, v_hist])
            stripe_feat /= np.linalg.norm(stripe_feat) + 1e-6
            feats.append(stripe_feat)

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.resize(np.hypot(gx, gy), (8, 12)).flatten()
        mag /= np.linalg.norm(mag) + 1e-6
        feats.append(mag)

        raw_384 = np.concatenate(feats).astype(np.float32)
        emb = raw_384 @ self._proj
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb /= norm
        return emb


class KalmanPersonTrack:
    """8 boyutlu sabit hızlı Kalman Filtresi ile tekil kişi izi durumu."""

    def __init__(
        self,
        track_id: int,
        bbox: Tuple[int, int, int, int],
        embedding: np.ndarray,
        confidence: float = 0.9,
        max_history: int = 600,
    ):
        self.track_id = track_id
        self.confidence = float(confidence)
        self.max_history = max_history

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(10.0, float(x2 - x1))
        h = max(20.0, float(y2 - y1))

        # Durum vektörü: [cx, cy, w, h, vx, vy, vw, vh]
        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(8, dtype=np.float64) * 10.0
        self.P[4:, 4:] *= 100.0

        self.F = np.eye(8, dtype=np.float64)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        self.H = np.eye(4, 8, dtype=np.float64)
        self.R = np.diag([4.0, 4.0, 10.0, 10.0])
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.25, 0.25, 0.05, 0.05])

        self.embedding = embedding.copy()
        self.age = 1
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0

        self.history: List[Tuple[int, int]] = [(int(round(cx)), int(round(cy)))]

    def predict(self) -> List[int]:
        """Kalman durumunu bir kare ilerletir ve tahmin edilen [xmin, ymin, xmax, ymax] kutusunu döndürür."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x[2] = max(10.0, self.x[2])
        self.x[3] = max(20.0, self.x[3])

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.to_bbox()

    def update(
        self,
        bbox: Tuple[int, int, int, int],
        embedding: np.ndarray,
        confidence: float = 0.9,
    ) -> None:
        """Eşleşen ölçüm ile Kalman durumunu ve Re-ID görünüm vektörünü günceller."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(10.0, float(x2 - x1))
        h = max(20.0, float(y2 - y1))
        z = np.array([cx, cy, w, h], dtype=np.float64)

        y_res = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y_res)
        self.P = (np.eye(8) - K @ self.H) @ self.P

        # Görünüm vektörünü üstel hareketli ortalama (EMA) ile güncelle
        alpha = 0.85
        updated_emb = alpha * self.embedding + (1.0 - alpha) * embedding
        norm = np.linalg.norm(updated_emb)
        if norm > 1e-6:
            self.embedding = updated_emb / norm

        self.confidence = float(confidence)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        curr_centroid = (int(round(self.x[0])), int(round(self.x[1])))
        self.history.append(curr_centroid)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

    def to_bbox(self) -> List[int]:
        """Güncel [xmin, ymin, xmax, ymax] tamsayı kutu koordinatlarını döndürür."""
        cx, cy, w, h = self.x[:4]
        x1 = int(round(cx - w / 2.0))
        y1 = int(round(cy - h / 2.0))
        x2 = int(round(cx + w / 2.0))
        y2 = int(round(cy + h / 2.0))
        return [x1, y1, x2, y2]

    @property
    def centroid(self) -> Tuple[int, int]:
        return (int(round(self.x[0])), int(round(self.x[1])))

    @property
    def velocity(self) -> Tuple[float, float]:
        return (float(self.x[4]), float(self.x[5]))


class ObjectTracker:
    """Kalman Filtresi ve Re-ID Görünüm Eşleştirmeli DeepSORT Çoklu Nesne Takipçisi."""

    def __init__(
        self,
        max_disappeared: int = 30,
        max_distance: float = 140.0,
        min_hits: int = 1,
        iou_threshold: float = 0.25,
        appearance_weight: float = 0.40,
        feature_dim: int = 512,
    ):
        self.next_object_id = 0
        self.max_disappeared = int(max_disappeared)

        if max_distance <= 1.0:
            self.iou_threshold = float(max_distance)
            self.max_distance = 140.0
        else:
            self.max_distance = float(max_distance)
            self.iou_threshold = float(iou_threshold)

        self.min_hits = int(min_hits)
        self.appearance_weight = float(appearance_weight)
        self.reid = ReIDAppearanceExtractor(feature_dim=feature_dim)

        self.tracks: Dict[int, KalmanPersonTrack] = {}
        self.objects: Dict[int, Tuple[int, int]] = {}
        self.disappeared: Dict[int, int] = {}

    def _sync_legacy_dicts(self) -> None:
        self.objects = {tid: trk.centroid for tid, trk in self.tracks.items()}
        self.disappeared = {
            tid: trk.time_since_update for tid, trk in self.tracks.items()
        }

    def register(
        self,
        bbox: Tuple[int, int, int, int],
        embedding: Optional[np.ndarray] = None,
        confidence: float = 0.9,
    ) -> int:
        """Yeni kişi izi kaydeder."""
        if len(bbox) == 2:
            cx, cy = bbox
            bbox = (cx - 25, cy - 50, cx + 25, cy + 50)

        if embedding is None:
            embedding = np.ones(self.reid.feature_dim, dtype=np.float32)
            embedding /= np.linalg.norm(embedding)

        tid = self.next_object_id
        self.tracks[tid] = KalmanPersonTrack(
            track_id=tid,
            bbox=bbox,
            embedding=embedding,
            confidence=confidence,
        )
        self.next_object_id += 1
        self._sync_legacy_dicts()
        return tid

    def deregister(self, object_id: int) -> None:
        """Kaybolan izi kimliğine göre siler."""
        self.tracks.pop(object_id, None)
        self._sync_legacy_dicts()

    def update(
        self,
        rects: List[Tuple[int, int, int, int]],
        frame: Optional[np.ndarray] = None,
        confidences: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Yeni tespitlerle DeepSORT izlerini günceller ve aktif iz listesini döndürür."""
        normalized_rects: List[Tuple[int, int, int, int]] = []
        extracted_confs: List[float] = []
        for idx, r in enumerate(rects):
            if len(r) >= 5:
                normalized_rects.append((int(r[0]), int(r[1]), int(r[2]), int(r[3])))
                extracted_confs.append(float(r[4]))
            else:
                normalized_rects.append((int(r[0]), int(r[1]), int(r[2]), int(r[3])))
                if confidences and idx < len(confidences):
                    extracted_confs.append(float(confidences[idx]))
                else:
                    extracted_confs.append(0.90)

        track_ids = list(self.tracks.keys())
        predicted_boxes = [self.tracks[tid].predict() for tid in track_ids]

        if len(normalized_rects) == 0:
            for tid in list(self.tracks.keys()):
                if self.tracks[tid].time_since_update > self.max_disappeared:
                    self.deregister(tid)
            self._sync_legacy_dicts()
            return self._format_outputs()

        det_boxes = np.array(normalized_rects, dtype=np.float32)
        det_centroids = np.column_stack(
            [(det_boxes[:, 0] + det_boxes[:, 2]) / 2.0, (det_boxes[:, 1] + det_boxes[:, 3]) / 2.0]
        )
        det_embeddings = np.array(
            [self.reid.extract(frame, b) for b in normalized_rects],
            dtype=np.float32,
        )

        if len(track_ids) == 0:
            for i, bbox in enumerate(normalized_rects):
                self.register(bbox, det_embeddings[i], extracted_confs[i])
            return self._format_outputs()

        trk_boxes = np.array(predicted_boxes, dtype=np.float32)
        trk_centroids = np.column_stack(
            [(trk_boxes[:, 0] + trk_boxes[:, 2]) / 2.0, (trk_boxes[:, 1] + trk_boxes[:, 3]) / 2.0]
        )
        trk_embeddings = np.array(
            [self.tracks[tid].embedding for tid in track_ids], dtype=np.float32
        )

        center_dist = dist.cdist(trk_centroids, det_centroids, metric="euclidean")
        norm_center_cost = np.clip(center_dist / max(self.max_distance, 1.0), 0.0, 2.0)

        iou_mat = compute_iou_matrix(trk_boxes, det_boxes)
        iou_cost = 1.0 - iou_mat

        cosine_sim = np.clip(trk_embeddings @ det_embeddings.T, -1.0, 1.0)
        appearance_cost = 0.5 * (1.0 - cosine_sim)

        if frame is not None:
            cost_matrix = (
                (1.0 - self.appearance_weight) * (0.55 * norm_center_cost + 0.45 * iou_cost)
                + self.appearance_weight * appearance_cost
            )
        else:
            cost_matrix = 0.65 * norm_center_cost + 0.35 * iou_cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_rows = set()
        matched_cols = set()

        for r, c in zip(row_ind, col_ind):
            if center_dist[r, c] <= self.max_distance or iou_mat[r, c] >= self.iou_threshold:
                tid = track_ids[r]
                self.tracks[tid].update(
                    normalized_rects[c],
                    det_embeddings[c],
                    extracted_confs[c],
                )
                matched_rows.add(r)
                matched_cols.add(c)

        for c in range(len(normalized_rects)):
            if c not in matched_cols:
                self.register(normalized_rects[c], det_embeddings[c], extracted_confs[c])

        for r, tid in enumerate(track_ids):
            if r not in matched_rows:
                if self.tracks[tid].time_since_update > self.max_disappeared:
                    self.deregister(tid)

        self._sync_legacy_dicts()
        return self._format_outputs()

    def _format_outputs(self) -> List[Dict[str, Any]]:
        """Aşağı akış modülleri için aktif izleri standart sözlük biçiminde döndürür."""
        outputs: List[Dict[str, Any]] = []
        for tid, trk in self.tracks.items():
            if trk.time_since_update <= 2 and (trk.hits >= self.min_hits or trk.age <= 3):
                outputs.append(
                    {
                        "track_id": tid,
                        "bbox": trk.to_bbox(),
                        "centroid": trk.centroid,
                        "confidence": trk.confidence,
                        "history": list(trk.history),
                        "velocity": trk.velocity,
                        "age": trk.age,
                        "hits": trk.hits,
                    }
                )
        return outputs


DeepSORTTracker = ObjectTracker
