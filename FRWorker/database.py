"""LanceDB helpers for face embeddings.

Replaces the SQLite backend. LanceDB is directory-based and uses an
append-only file format that allows multiple concurrent writers safely.
"""
import hashlib
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyarrow as pa
import lancedb

VECTOR_DIM = 512
TABLE_NAME = "faces"

_schema = pa.schema([
    pa.field("id",       pa.int64()),
    pa.field("filename", pa.utf8()),
    pa.field("vector",   pa.list_(pa.float32(), VECTOR_DIM)),
])


def open_table(db_path: str):
    """Open (or create) the LanceDB faces table.
    Safe to call concurrently from multiple processes.
    Returns a LanceDB Table object.
    """
    db = lancedb.connect(db_path)
    return db.create_table(TABLE_NAME, schema=_schema, exist_ok=True)


def insert_face(table, filename: str, embedding: List[float]) -> int:
    """Append one face record. Safe to call from multiple concurrent workers."""
    row_id = int(hashlib.sha1(filename.encode()).hexdigest()[:10], 16)
    vec = np.array(embedding, dtype=np.float32)
    table.add([{"id": row_id, "filename": filename, "vector": vec.tolist()}])
    return row_id


def find_nearest(table, query_embedding: List[float], top_k: int = 5) -> List[Tuple[int, str, float]]:
    """Return [(id, filename, similarity)] using cosine similarity.
    LanceDB cosine metric returns distance = 1 - cosine_sim, so we invert it.
    """
    if len(table) == 0:
        return []
    vec = np.array(query_embedding, dtype=np.float32)
    # Use L2 distance on normalized vectors: cosine_sim = 1 - (L2_dist² / 2)
    # This avoids a LanceDB cosine distance bug with certain vector distributions.
    rows = (
        table.search(vec)
             .metric("l2")
             .limit(top_k)
             .to_list()
    )
    return [
        (int(r["id"]), r["filename"], round(1.0 - r["_distance"] / 2.0, 4))
        for r in rows
    ]
