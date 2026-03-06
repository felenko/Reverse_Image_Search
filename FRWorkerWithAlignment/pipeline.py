"""End-to-end face extraction pipeline.

FacePipeline orchestrates:
  1. RetinaFace detection         (face_detector.py)
  2. Yaw-based alignment          (face_aligner.py)
     - standard 5-pt similarity  (|yaw| < 20°)
     - full affine 5-pt           (20° <= |yaw| <= 45°)
     - 3D frontalization + 5-pt  (|yaw| > 45°)
  3. ArcFace embedding            (embedding_model.py)

Usage:
    pipe = FacePipeline(device='cpu')
    results = pipe.process_bgr(img_bgr)
    for r in results:
        print(r.yaw, r.alignment_method, r.confidence, r.embedding[:4])
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Pre-processing helpers
# ---------------------------------------------------------------------------

# RetinaFace needs context around the face.  Images where the face fills the
# entire frame (tight crops, synthetic datasets) fail detection without margin.
_PAD_MIN_DIM   = 120    # pad any image smaller than this (pixels per side)
_PAD_FRACTION  = 0.55   # add this fraction of w/h on each side


def _pad_for_detection(img_bgr: np.ndarray) -> np.ndarray:
    """Add border padding to images that are too tightly cropped for RetinaFace.

    When the face fills the entire image (common in synthetic/pose datasets),
    RetinaFace scores drop below threshold.  Adding ~55% context on each side
    restores detection without affecting alignment (padded image is used
    throughout the pipeline).
    """
    h, w = img_bgr.shape[:2]
    if h >= _PAD_MIN_DIM and w >= _PAD_MIN_DIM:
        return img_bgr
    py = max(int(h * _PAD_FRACTION), 16)
    px = max(int(w * _PAD_FRACTION), 16)
    return cv2.copyMakeBorder(img_bgr, py, py, px, px, cv2.BORDER_REPLICATE)

from face_aligner    import AlignmentMethod, FaceAligner
from face_detector   import FaceDetection, RetinaFaceDetector
from embedding_model import ArcFaceExtractor
from frontalizer     import Frontalizer


@dataclass
class FaceResult:
    filename:         str                      # original image path
    embedding:        Optional[List[float]]    # 512-d unit-norm ArcFace vector
    yaw:              float                    # estimated yaw in degrees
    alignment_method: str                      # AlignmentMethod value string
    confidence:       float                    # det_score * alignment_multiplier
    det_score:        float                    # raw RetinaFace detection score
    bbox:             Optional[np.ndarray] = field(default=None, repr=False)


class FacePipeline:
    """Reusable pipeline; holds lazy-loaded models as instance state."""

    def __init__(
        self,
        device: str             = "cpu",
        det_size: tuple         = (640, 640),
        det_thresh: float       = 0.5,
        tddfa_root: Optional[str] = None,
    ):
        self._device     = device
        self._det_size   = det_size
        self._det_thresh = det_thresh
        self._tddfa_root = tddfa_root

        self._detector    = None
        self._aligner     = None
        self._embedder    = None

    # ------------------------------------------------------------------
    # Lazy model accessors
    # ------------------------------------------------------------------

    @property
    def detector(self) -> RetinaFaceDetector:
        if self._detector is None:
            self._detector = RetinaFaceDetector(
                det_size=self._det_size,
                device=self._device,
                det_thresh=self._det_thresh,
            )
        return self._detector

    @property
    def aligner(self) -> FaceAligner:
        if self._aligner is None:
            frontalizer   = Frontalizer(tddfa_root=self._tddfa_root)
            self._aligner = FaceAligner(
                frontalizer=frontalizer,
                detector=self.detector,   # reuse loaded detector for re-detection
            )
        return self._aligner

    @property
    def embedder(self) -> ArcFaceExtractor:
        if self._embedder is None:
            self._embedder = ArcFaceExtractor(device=self._device)
        return self._embedder

    # ------------------------------------------------------------------
    # Public processing methods
    # ------------------------------------------------------------------

    def process_bgr(
        self,
        img_bgr: np.ndarray,
        filename: str = "",
        max_faces: int = 1,
    ) -> List[FaceResult]:
        """Process a BGR numpy image.

        Args:
            img_bgr   : OpenCV-style BGR uint8 array
            filename  : label stored in results (e.g. relative image path)
            max_faces : maximum number of faces to process (default 1 = largest)

        Returns:
            List of FaceResult; empty if no face detected.
        """
        img_bgr    = _pad_for_detection(img_bgr)
        detections = self.detector.detect(img_bgr)
        if not detections:
            return []

        results: List[FaceResult] = []
        for det in detections[:max_faces]:
            crop, method, conf_mult = self.aligner.align(img_bgr, det)
            if crop is None:
                continue

            confidence = float(det.det_score) * conf_mult
            emb        = self.embedder.get_embedding(crop)

            results.append(FaceResult(
                filename         = filename,
                embedding        = emb,
                yaw              = det.yaw,
                alignment_method = method.value,
                confidence       = round(confidence, 4),
                det_score        = round(det.det_score, 4),
                bbox             = det.bbox,
            ))

        return results

    def process_path(
        self,
        path: str,
        max_faces: int = 1,
    ) -> List[FaceResult]:
        """Load image from disk and process it."""
        img = Image.open(path).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return self.process_bgr(img_bgr, filename=path, max_faces=max_faces)
