"""ArcFace embedding extractor using insightface's w600k_r50.onnx model.

Accepts a pre-aligned 112x112 BGR numpy array (uint8 or float32) and returns
a unit-normalised 512-d numpy vector.

The recognition model is loaded lazily and shared across calls.
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np


class ArcFaceExtractor:
    """Wraps insightface's ArcFace ONNX recognition model.

    The model is separate from the detector: we pass our own aligned crops
    rather than letting insightface do internal alignment.
    """

    def __init__(self, device: str = "cpu"):
        self._device   = device
        self._rec      = None    # insightface ArcFaceONNX model

    # ------------------------------------------------------------------
    @property
    def model(self):
        if self._rec is None:
            self._rec = self._load_rec_model()
        return self._rec

    # ------------------------------------------------------------------
    def _load_rec_model(self):
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "insightface is required. Install: pip install insightface onnxruntime"
            )

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self._device == "cuda"
            else ["CPUExecutionProvider"]
        )
        ctx_id = 0 if self._device == "cuda" else -1

        # FaceAnalysis asserts 'detection' is present; include it even though
        # we only need the recognition model here.
        fa = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=providers,
        )
        fa.prepare(ctx_id=ctx_id, det_size=(112, 112))

        if "recognition" not in fa.models:
            raise RuntimeError(
                "buffalo_l recognition model not found. "
                "Ensure insightface models are downloaded (~/.insightface/models/buffalo_l/)."
            )
        rec = fa.models["recognition"]
        print(f"[ArcFaceExtractor] Loaded {rec.__class__.__name__} on {self._device}.")
        return rec

    # ------------------------------------------------------------------
    def get_embedding(self, aligned_bgr_112: np.ndarray) -> Optional[List[float]]:
        """Extract ArcFace embedding from a 112x112 BGR crop.

        Args:
            aligned_bgr_112: uint8 or float32 BGR image, shape (112, 112, 3)

        Returns:
            512-d list[float] unit-normalised embedding, or None on failure.
        """
        if aligned_bgr_112 is None:
            return None

        img = aligned_bgr_112
        if img.dtype != np.uint8:
            img = np.clip(img * 255, 0, 255).astype(np.uint8)

        if img.shape[:2] != (112, 112):
            img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)

        try:
            # get_feat expects a list of BGR 112x112 uint8 arrays
            # insightface normalises internally: (pixel - 127.5) / input_std
            emb = self.model.get_feat([img])   # (1, 512)
            vec = np.array(emb[0], dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
            return vec.tolist()
        except Exception as exc:
            import traceback
            print(f"[ArcFaceExtractor] get_feat failed: {exc}")
            traceback.print_exc()
            return None
