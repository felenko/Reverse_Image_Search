"""LanceDB helpers for face embeddings with alignment metadata.

Extended schema over the base FRWorker:
  - yaw              : estimated yaw angle in degrees at index time
  - alignment_method : 'sim5pt' | 'affine5pt' | 'frontal_3ddfa' | 'frontal_pnp'
  - confidence       : det_score * alignment_confidence_multiplier  (0–1)
"""
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyarrow as pa
import lancedb

VECTOR_DIM  = 512
TABLE_NAME  = "faces_aligned"

_schema = pa.schema([
    pa.field("id",               pa.int64()),
    pa.field("filename",         pa.utf8()),
    pa.field("vector",           pa.list_(pa.float32(), VECTOR_DIM)),
    pa.field("yaw",              pa.float32()),
    pa.field("alignment_method", pa.utf8()),
    pa.field("confidence",       pa.float32()),
])


def open_table(db_path: str):
    """Open (or create) the LanceDB aligned-faces table.
    Safe to call concurrently from multiple processes.
    """
    db = lancedb.connect(db_path)
    return db.create_table(TABLE_NAME, schema=_schema, exist_ok=True)


def insert_face(
    table,
    filename: str,
    embedding: List[float],
    yaw: float = 0.0,
    alignment_method: str = "sim5pt",
    confidence: float = 1.0,
) -> int:
    """Append one face record. Safe to call from multiple concurrent workers."""
    row_id = int(hashlib.sha1(filename.encode()).hexdigest()[:10], 16)
    vec = np.array(embedding, dtype=np.float32)
    table.add([{
        "id":               row_id,
        "filename":         filename,
        "vector":           vec.tolist(),
        "yaw":              float(yaw),
        "alignment_method": alignment_method,
        "confidence":       float(np.clip(confidence, 0.0, 1.0)),
    }])
    return row_id


def find_nearest(
    table,
    query_embedding: List[float],
    top_k: int = 5,
    min_confidence: float = 0.0,
) -> List[Tuple[int, str, float, str, float]]:
    """Return [(id, filename, similarity, alignment_method, confidence)].

    Uses L2 distance on unit-normalized vectors (equivalent to cosine similarity).
    Optionally filters by minimum indexed confidence.
    """
    if len(table) == 0:
        return []
    vec = np.array(query_embedding, dtype=np.float32)
    rows = (
        table.search(vec)
             .metric("l2")
             .limit(top_k * 4 if min_confidence > 0 else top_k)
             .to_list()
    )
    results = []
    for r in rows:
        sim = round(1.0 - r["_distance"] / 2.0, 4)
        if r["confidence"] >= min_confidence:
            results.append((
                int(r["id"]),
                r["filename"],
                sim,
                r["alignment_method"],
                round(float(r["confidence"]), 4),
            ))
    return results[:top_k]
