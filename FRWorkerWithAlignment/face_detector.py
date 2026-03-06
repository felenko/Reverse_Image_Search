"""RetinaFace-based face detector with 106-point landmarks and pose estimation.

Uses insightface's buffalo_l model pack:
  - det_10g.onnx     : RetinaFace detector  -> bbox, 5-pt kps, det_score
  - 2d106det.onnx    : 106-point 2D landmark detector -> dense landmarks + pose

Pose (yaw/pitch/roll) is either taken from insightface's calc_pose() or
estimated from 5-pt keypoints using a geometric heuristic when unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class FaceDetection:
    bbox:             np.ndarray          # (4,) float32  [x1, y1, x2, y2]
    kps_5pt:          np.ndarray          # (5, 2) float32 standard keypoints
    det_score:        float
    yaw:              float               # degrees; positive = turned right
    pitch:            float               # degrees; positive = looking up
    roll:             float               # degrees; positive = tilted right
    kps_106:          Optional[np.ndarray] = field(default=None)  # (106, 2)


def _estimate_yaw_from_5pt(kps: np.ndarray) -> float:
    """Geometric yaw estimate from 5-point landmarks.

    Uses the distance asymmetry between the nose tip and each eye.
    Positive yaw = face turned right.
    """
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    d_l = float(np.linalg.norm(nose - left_eye))
    d_r = float(np.linalg.norm(nose - right_eye))
    total = d_l + d_r
    if total < 1e-6:
        return 0.0
    # asymmetry ranges from -1 (full right turn) to +1 (full left turn)
    asymmetry = (d_l - d_r) / total
    # Empirical calibration: ratio ~0.33 corresponds to ~45°
    return float(np.degrees(np.arctan(asymmetry * 2.5)))


class RetinaFaceDetector:
    """Thin wrapper around insightface FaceAnalysis for detection + landmarks."""

    def __init__(
        self,
        det_size: tuple = (640, 640),
        device: str = "cpu",
        det_thresh: float = 0.5,
    ):
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "insightface is required. Install: pip install insightface onnxruntime"
            )

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        ctx_id = 0 if device == "cuda" else -1

        # Load detection + landmark models.
        # landmark_3d_68 provides accurate yaw/pitch/roll via face.pose.
        # landmark_2d_106 provides dense 2D landmarks.
        self._fa = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "landmark_2d_106", "landmark_3d_68"],
            providers=providers,
        )
        self._fa.prepare(ctx_id=ctx_id, det_size=det_size)
        self._det_thresh = det_thresh

    def detect(self, img_bgr: np.ndarray) -> List[FaceDetection]:
        """Run RetinaFace + landmark detection on a BGR image.

        Returns a list of FaceDetection objects sorted by det_score descending.
        """
        faces = self._fa.get(img_bgr)
        results: List[FaceDetection] = []

        for f in faces:
            if float(f.det_score) < self._det_thresh:
                continue

            kps_5pt = np.array(f.kps, dtype=np.float32)  # (5, 2)
            kps_106 = (
                np.array(f.landmark_2d_106, dtype=np.float32)
                if hasattr(f, "landmark_2d_106") and f.landmark_2d_106 is not None
                else None
            )

            # Prefer insightface pose (from landmark_3d_68 — accurate to ±90°).
            # Fall back to geometric heuristic if unavailable.
            if hasattr(f, "pose") and f.pose is not None:
                pose  = np.asarray(f.pose).flatten()
                # insightface returns [pitch, yaw, roll] in degrees
                pitch = float(pose[0]) if len(pose) > 0 else 0.0
                yaw   = float(pose[1]) if len(pose) > 1 else _estimate_yaw_from_5pt(kps_5pt)
                roll  = float(pose[2]) if len(pose) > 2 else 0.0
            else:
                yaw   = _estimate_yaw_from_5pt(kps_5pt)
                pitch = 0.0
                roll  = 0.0
                if abs(yaw) > 25:
                    # Heuristic saturates beyond ~68°; flag so callers know
                    pass  # alignment routing will still use the magnitude correctly

            results.append(FaceDetection(
                bbox=np.array(f.bbox, dtype=np.float32),
                kps_5pt=kps_5pt,
                det_score=float(f.det_score),
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                kps_106=kps_106,
            ))

        results.sort(key=lambda x: x.det_score, reverse=True)
        return results
