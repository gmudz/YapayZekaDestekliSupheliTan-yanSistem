"""Milvus Vektör Veritabanı İzleme Listesi Yöneticisi (Bulut Katmanı — TÜBİTAK 2209-A).

Milvus Standalone sunucusuna (19530 portu; yerel yoksa WireGuard merkez 10.66.66.1:19530)
bağlanarak 512 boyutlu ArcFace gömme vektörlerini Kosinüs Benzerliği (COSINE) metriği ile
saklar ve arar. Aynı zamanda kesintisiz çalışma için yerel JSON yansısını senkronize tutar.
"""

import os
import json
import time
import uuid
import numpy as np
from typing import Dict, List, Optional, Any


class MilvusWatchlistDB:
    """512-D ArcFace İzleme Listesi araması için Milvus Vektör Veritabanı istemcisi."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 19530,
        collection_name: str = "tubitak_watchlist_faces",
        embedding_dim: int = 512,
        local_fallback_path: str = "data/watchlist_store.json",
    ):
        self.host = host
        self.port = int(port)
        self.collection_name = collection_name
        self.embedding_dim = int(embedding_dim)

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.isabs(local_fallback_path):
            local_fallback_path = os.path.join(base_dir, local_fallback_path)
        self.local_fallback_path = local_fallback_path
        os.makedirs(os.path.dirname(self.local_fallback_path), exist_ok=True)

        self.milvus_connected = False
        self.collection = None
        self.records: Dict[str, Dict[str, Any]] = {}

        self._load_local_store()
        self.connect_milvus()

    def _load_local_store(self) -> None:
        if os.path.exists(self.local_fallback_path):
            try:
                with open(self.local_fallback_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.records = data
            except Exception as e:
                print(f"[MilvusVeritabani] Yerel yedek yükleme uyarısı: {e}")

    def _save_local_store(self) -> None:
        try:
            with open(self.local_fallback_path, "w", encoding="utf-8") as f:
                json.dump(self.records, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[MilvusVeritabani] Yerel yedek kaydetme uyarısı: {e}")

    def connect_milvus(self) -> bool:
        """Milvus Standalone sunucusuna bağlanır ve 512-D COSINE koleksiyonunu hazırlar."""
        try:
            from pymilvus import (
                connections,
                utility,
                FieldSchema,
                CollectionSchema,
                DataType,
                Collection,
            )

            try:
                connections.connect(
                    alias="default",
                    host=self.host,
                    port=str(self.port),
                    timeout=2.0,
                )
            except Exception:
                connections.connect(
                    alias="default",
                    host="10.66.66.1",
                    port=str(self.port),
                    timeout=2.0,
                )
                self.host = "10.66.66.1"

            if not utility.has_collection(self.collection_name):
                fields = [
                    FieldSchema(
                        name="person_id",
                        dtype=DataType.VARCHAR,
                        is_primary=True,
                        max_length=64,
                    ),
                    FieldSchema(
                        name="name",
                        dtype=DataType.VARCHAR,
                        max_length=128,
                    ),
                    FieldSchema(
                        name="risk_level",
                        dtype=DataType.VARCHAR,
                        max_length=32,
                    ),
                    FieldSchema(
                        name="demographic_group",
                        dtype=DataType.VARCHAR,
                        max_length=64,
                    ),
                    FieldSchema(
                        name="embedding",
                        dtype=DataType.FLOAT_VECTOR,
                        dim=self.embedding_dim,
                    ),
                ]
                schema = CollectionSchema(
                    fields=fields,
                    description="TÜBİTAK 2209-A ArcFace 512-D İzleme Listesi Koleksiyonu",
                )
                self.collection = Collection(
                    name=self.collection_name, schema=schema
                )
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128},
                }
                self.collection.create_index(
                    field_name="embedding", index_params=index_params
                )
            else:
                self.collection = Collection(self.collection_name)

            self.collection.load()
            self.milvus_connected = True

            if self.collection.num_entities == 0 and len(self.records) > 0:
                self._sync_all_to_milvus()

            return True
        except Exception:
            self.milvus_connected = False
            self.collection = None
            return False

    def _sync_all_to_milvus(self) -> None:
        if not self.milvus_connected or self.collection is None:
            return
        try:
            pids, names, risks, groups, embs = [], [], [], [], []
            for pid, rec in self.records.items():
                pids.append(str(pid))
                names.append(str(rec.get("name", "Bilinmiyor")))
                risks.append(str(rec.get("risk_level", "YÜKSEK")))
                groups.append(str(rec.get("demographic_group", "Orta_Dogulu")))
                vec = np.array(rec["embedding"], dtype=np.float32)
                vec /= np.linalg.norm(vec) + 1e-8
                embs.append(vec.tolist())
            if pids:
                self.collection.insert([pids, names, risks, groups, embs])
                self.collection.flush()
                self.collection.load()
        except Exception as e:
            print(f"[MilvusVeritabani] Senkronizasyon uyarısı: {e}")

    def enroll_identity(
        self,
        name: str,
        embedding: np.ndarray,
        risk_level: str = "YÜKSEK",
        demographic_group: str = "Orta_Dogulu",
        person_id: Optional[str] = None,
        image_b64: Optional[str] = None,
        notes: str = "",
    ) -> Dict[str, Any]:
        """512 boyutlu ArcFace vektörünü Milvus ve yerel izleme listesi deposuna kaydeder."""
        pid = person_id or f"WL-{uuid.uuid4().hex[:8].upper()}"
        vec = np.asarray(embedding, dtype=np.float32).flatten()
        if len(vec) != self.embedding_dim:
            raise ValueError(
                f"{self.embedding_dim} boyutlu vektör bekleniyordu, gelen boyut: {len(vec)}"
            )
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm

        record = {
            "person_id": pid,
            "name": name,
            "risk_level": risk_level,
            "demographic_group": demographic_group,
            "embedding": vec.tolist(),
            "image_b64": image_b64,
            "notes": notes,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.records[pid] = record
        self._save_local_store()

        if not self.milvus_connected:
            self.connect_milvus()

        if self.milvus_connected and self.collection is not None:
            try:
                self.collection.delete(f'person_id == "{pid}"')
                self.collection.insert(
                    [
                        [pid],
                        [name],
                        [risk_level],
                        [demographic_group],
                        [vec.tolist()],
                    ]
                )
                self.collection.flush()
                self.collection.load()
            except Exception as e:
                print(f"[MilvusVeritabani] Milvus ekleme uyarısı: {e}")

        return {
            "person_id": pid,
            "name": name,
            "risk_level": risk_level,
            "demographic_group": demographic_group,
            "backend": "Milvus Standalone (Canlı)" if self.milvus_connected else "Yerel Kosinüs Motoru",
        }

    def search_watchlist(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        threshold: float = 0.52,
    ) -> Dict[str, Any]:
        """Kosinüs Benzerliği (COSINE) kullanarak Milvus İzleme Listesinde arama yapar."""
        q_vec = np.asarray(query_embedding, dtype=np.float32).flatten()
        norm = np.linalg.norm(q_vec)
        if norm > 1e-8:
            q_vec = q_vec / norm

        start_t = time.perf_counter()
        matches: List[Dict[str, Any]] = []
        used_backend = "Yerel Kosinüs Vektör Motoru"

        if not self.milvus_connected:
            self.connect_milvus()

        if self.milvus_connected and self.collection is not None:
            try:
                search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
                res = self.collection.search(
                    data=[q_vec.tolist()],
                    anns_field="embedding",
                    param=search_params,
                    limit=top_k,
                    output_fields=["person_id", "name", "risk_level", "demographic_group"],
                )
                used_backend = "Milvus Standalone v2.4"
                if res and len(res[0]) > 0:
                    for hit in res[0]:
                        sim = float(hit.score)
                        pid = hit.entity.get("person_id")
                        local_rec = self.records.get(pid, {})
                        matches.append(
                            {
                                "person_id": pid,
                                "name": hit.entity.get("name") or local_rec.get("name", "Bilinmiyor"),
                                "risk_level": hit.entity.get("risk_level") or local_rec.get("risk_level", "YÜKSEK"),
                                "demographic_group": hit.entity.get("demographic_group")
                                or local_rec.get("demographic_group", "Orta_Dogulu"),
                                "cosine_similarity": round(sim, 4),
                                "is_match": sim >= threshold,
                                "image_b64": local_rec.get("image_b64"),
                            }
                        )
            except Exception:
                matches = []

        if not matches and len(self.records) > 0:
            scored = []
            for pid, rec in self.records.items():
                ref_vec = np.asarray(rec["embedding"], dtype=np.float32)
                ref_norm = np.linalg.norm(ref_vec)
                if ref_norm > 1e-8:
                    ref_vec = ref_vec / ref_norm
                sim = float(np.dot(q_vec, ref_vec))
                scored.append(
                    {
                        "person_id": pid,
                        "name": rec.get("name", "Bilinmiyor"),
                        "risk_level": rec.get("risk_level", "YÜKSEK"),
                        "demographic_group": rec.get("demographic_group", "Orta_Dogulu"),
                        "cosine_similarity": round(sim, 4),
                        "is_match": sim >= threshold,
                        "image_b64": rec.get("image_b64"),
                    }
                )
            scored.sort(key=lambda x: x["cosine_similarity"], reverse=True)
            matches = scored[:top_k]

        latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
        best_match = matches[0] if matches else None
        is_watchlist_hit = bool(best_match and best_match["cosine_similarity"] >= threshold)

        return {
            "is_watchlist_hit": is_watchlist_hit,
            "best_match": best_match if is_watchlist_hit else None,
            "top_candidates": matches,
            "threshold": threshold,
            "metric": "COSINE",
            "backend": used_backend,
            "search_latency_ms": latency_ms,
        }

    def list_identities(self) -> List[Dict[str, Any]]:
        """İzleme listesinde kayıtlı tüm profilleri döndürür."""
        out = []
        for pid, rec in self.records.items():
            out.append(
                {
                    "person_id": pid,
                    "name": rec.get("name"),
                    "risk_level": rec.get("risk_level"),
                    "demographic_group": rec.get("demographic_group"),
                    "notes": rec.get("notes", ""),
                    "created_at": rec.get("created_at", ""),
                    "image_b64": rec.get("image_b64"),
                    "embedding_dim": len(rec.get("embedding", [])),
                }
            )
        return out

    def delete_identity(self, person_id: str) -> bool:
        existed = person_id in self.records
        self.records.pop(person_id, None)
        self._save_local_store()
        if self.milvus_connected and self.collection is not None:
            try:
                self.collection.delete(f'person_id == "{person_id}"')
                self.collection.flush()
            except Exception:
                pass
        return existed

    def get_status(self) -> Dict[str, Any]:
        if not self.milvus_connected:
            self.connect_milvus()
        return {
            "milvus_connected": self.milvus_connected,
            "host": self.host,
            "port": self.port,
            "collection_name": self.collection_name,
            "embedding_dim": self.embedding_dim,
            "metric_type": "COSINE",
            "watchlist_count": len(self.records),
            "backend": "Milvus Standalone v2.4 (COSINE)"
            if self.milvus_connected
            else "Yerel Kosinüs Vektör Motoru (Milvus Yedeği)",
        }
