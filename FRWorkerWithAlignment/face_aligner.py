"""Yaw-based face alignment producing 112x112 BGR crops for ArcFace.

Strategy:
  |yaw| <  20°  → STANDARD_5PT   : similarity transform from 5-pt kps
  |yaw| 20-45°  → AFFINE_5PT     : full affine transform (handles foreshortening)
  |yaw| >  45°  → FRONTAL_*      : frontalize first, re-detect, then STANDARD_5PT
                                    confidence multiplier applied from Frontalizer

All output crops are 112x112 BGR float32 arrays suitable for ArcFace embedding.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np

from face_detector import FaceDetection

# ---------------------------------------------------------------------------
# ArcFace canonical landmark positions on a 112x112 face crop
# [left_eye, right_eye, nose_tip, mouth_left, mouth_right]
# ---------------------------------------------------------------------------
ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

OUTPUT_SIZE = 112


class AlignmentMethod(str, Enum):
    STANDARD_5PT  = "sim5pt"       # similarity transform  (best quality, frontal)
    AFFINE_5PT    = "affine5pt"    # full affine transform  (moderate yaw)
    FRONTAL_3DDFA = "frontal_3ddfa"
    FRONTAL_PNP   = "frontal_pnp"


# ---------------------------------------------------------------------------
# Low-level transform helpers
# ---------------------------------------------------------------------------

def _warp_to_112(img_bgr: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply 2x3 affine matrix M and return 112x112 BGR crop."""
    return cv2.warpAffine(
        img_bgr, M, (OUTPUT_SIZE, OUTPUT_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _similarity_transform(src_pts: np.ndarray) -> np.ndarray:
    """Compute 2x3 similarity (4-DOF) transform: src_pts -> ARCFACE_DST.

    Uses cv2.estimateAffinePartial2D which fits scale + rotation + translation.
    """
    M, _ = cv2.estimateAffinePartial2D(
        src_pts, ARCFACE_DST,
        method=cv2.LMEDS,
        maxIters=1000,
    )
    if M is None:
        # Fallback: direct least-squares from first 3 points
        M = cv2.getAffineTransform(src_pts[:3], ARCFACE_DST[:3])
    return M.astype(np.float32)


def _affine_transform(src_pts: np.ndarray) -> np.ndarray:
    """Compute 2x3 full affine (6-DOF) transform: src_pts -> ARCFACE_DST.

    6-DOF allows independent x/y scaling to compensate for foreshortening.
    Uses RANSAC for robustness.
    """
    M, _ = cv2.estimateAffine2D(
        src_pts, ARCFACE_DST,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
    )
    if M is None:
        M = _similarity_transform(src_pts)
    return M.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class FaceAligner:
    """Routes each face detection to the appropriate alignment strategy."""

    YAW_THRESH_LOW  = 20.0   # degrees: below this -> similarity transform
    YAW_THRESH_HIGH = 45.0   # degrees: above this -> frontalization

    def __init__(self, frontalizer=None, detector=None):
        """
        Args:
            frontalizer: optional Frontalizer instance.  If None, large-yaw
                         faces are aligned with affine_5pt at reduced confidence.
            detector: optional RetinaFaceDetector instance reused for re-detection
                      after frontalization.  If None, re-detection is skipped.
        """
        self._frontalizer = frontalizer
        self._detector    = detector

    # ------------------------------------------------------------------
    def align(
        self,
        img_bgr: np.ndarray,
        detection: FaceDetection,
    ) -> Tuple[Optional[np.ndarray], AlignmentMethod, float]:
        """Align a detected face to 112x112 ArcFace canonical space.

        Returns:
            crop_112    : BGR 112x112 uint8 numpy array, or None on failure
            method      : AlignmentMethod used
            conf_mult   : confidence multiplier in (0, 1] to apply to det_score
        """
        abs_yaw = abs(detection.yaw)

        if abs_yaw < self.YAW_THRESH_LOW:
            return self._align_sim5pt(img_bgr, detection)

        if abs_yaw <= self.YAW_THRESH_HIGH:
            return self._align_affine5pt(img_bgr, detection)

        # Large yaw: attempt frontalization
        return self._align_frontal(img_bgr, detection)

    # ------------------------------------------------------------------
    def _align_sim5pt(
        self,
        img_bgr: np.ndarray,
        det: FaceDetection,
    ) -> Tuple[Optional[np.ndarray], AlignmentMethod, float]:
        M = _similarity_transform(det.kps_5pt)
        return _warp_to_112(img_bgr, M), AlignmentMethod.STANDARD_5PT, 1.0

    # ------------------------------------------------------------------
    def _align_affine5pt(
        self,
        img_bgr: np.ndarray,
        det: FaceDetection,
    ) -> Tuple[Optional[np.ndarray], AlignmentMethod, float]:
        """Full affine from 5-pt kps; compensates for lateral foreshortening."""
        M = _affine_transform(det.kps_5pt)
        # Slight confidence reduction because a 6-DOF warp can introduce artifacts
        conf = 1.0 - 0.15 * ((abs(det.yaw) - self.YAW_THRESH_LOW) /
                              (self.YAW_THRESH_HIGH - self.YAW_THRESH_LOW))
        return _warp_to_112(img_bgr, M), AlignmentMethod.AFFINE_5PT, max(conf, 0.75)

    # ------------------------------------------------------------------
    def _align_frontal(
        self,
        img_bgr: np.ndarray,
        det: FaceDetection,
    ) -> Tuple[Optional[np.ndarray], AlignmentMethod, float]:
        """Frontalize, re-detect, then apply standard 5-pt alignment."""
        if self._frontalizer is None:
            # No frontalizer: degrade to affine with lower confidence
            M = _affine_transform(det.kps_5pt)
            return _warp_to_112(img_bgr, M), AlignmentMethod.AFFINE_5PT, 0.55

        front_img, base_conf = self._frontalizer.frontalize(
            img_bgr, det.bbox, det.kps_5pt
        )
        if front_img is None:
            # Frontalization failed: fall back to affine at low confidence
            M = _affine_transform(det.kps_5pt)
            return _warp_to_112(img_bgr, M), AlignmentMethod.AFFINE_5PT, 0.50

        method = (
            AlignmentMethod.FRONTAL_3DDFA
            if self._frontalizer.backend == "3ddfa_v2"
            else AlignmentMethod.FRONTAL_PNP
        )

        # Re-detect face in frontalized image to get fresh landmarks
        re_det = self._re_detect(front_img, det)
        if re_det is None:
            # Re-detection failed on frontalized image; use direct 5-pt on original
            M = _affine_transform(det.kps_5pt)
            return _warp_to_112(img_bgr, M), method, base_conf * 0.8

        M = _similarity_transform(re_det.kps_5pt)
        return _warp_to_112(front_img, M), method, base_conf

    # ------------------------------------------------------------------
    def _re_detect(
        self,
        img_bgr: np.ndarray,
        original_det: FaceDetection,
    ) -> Optional[FaceDetection]:
        """Re-run detection on frontalized image; returns best matching face."""
        if self._detector is None:
            return None
        try:
            faces = self._detector.detect(img_bgr)
            if not faces:
                return None
            return _best_overlap(faces, original_det.bbox)
        except Exception:
            return None


def _best_overlap(faces: list, ref_bbox: np.ndarray) -> Optional[FaceDetection]:
    """Return the FaceDetection whose bbox best overlaps ref_bbox."""
    best, best_iou = None, -1.0
    rx1, ry1, rx2, ry2 = ref_bbox[:4]
    for f in faces:
        fx1, fy1, fx2, fy2 = f.bbox[:4]
        ix1, iy1 = max(rx1, fx1), max(ry1, fy1)
        ix2, iy2 = min(rx2, fx2), min(ry2, fy2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = ((rx2 - rx1) * (ry2 - ry1) + (fx2 - fx1) * (fy2 - fy1) - inter)
        iou   = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best, best_iou = f, iou
    return best
