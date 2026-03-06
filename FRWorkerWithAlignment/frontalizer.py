"""Face frontalization for large yaw angles (|yaw| > 45°).

Two backends (tried in order):

  1. 3DDFA_V2  – full 3D morphable model frontalization.
     Confidence multiplier: 0.82
     Requires manual install; see README.md.

  2. PnP-homography fallback  – estimates head pose via cv2.solvePnP and applies
     a perspective-correcting homography.  No extra dependencies beyond OpenCV.
     Confidence multiplier: 0.62

After frontalization, call detect() again on the returned image to obtain fresh
5-point landmarks for standard alignment.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Generic 3D face model: 5 landmark positions in mm (rough average adult)
# Order: left_eye_outer, right_eye_outer, nose_tip, mouth_left, mouth_right
# ---------------------------------------------------------------------------
_FACE_3D = np.array([
    [-43.0,  32.7, -26.0],
    [ 43.0,  32.7, -26.0],
    [  0.0,   0.0,   0.0],
    [-28.9, -28.9, -24.1],
    [ 28.9, -28.9, -24.1],
], dtype=np.float64)


def _camera_matrix(img_shape: tuple) -> np.ndarray:
    h, w = img_shape[:2]
    f = float(w)          # focal length estimate = image width
    return np.array([[f, 0, w / 2.0],
                     [0, f, h / 2.0],
                     [0, 0, 1.0   ]], dtype=np.float64)


def _pnp_frontalize(
    img_bgr: np.ndarray,
    kps_5pt: np.ndarray,
) -> Tuple[Optional[np.ndarray], float]:
    """Perspective-correct warp that roughly de-rotates the yaw component.

    Returns (warped_image, confidence_multiplier).
    """
    cam  = _camera_matrix(img_bgr.shape)
    dist = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        _FACE_3D,
        kps_5pt.astype(np.float64),
        cam, dist,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok:
        return None, 0.0

    R, _ = cv2.Rodrigues(rvec)
    t    = tvec.flatten()

    def _project(R_mat: np.ndarray) -> np.ndarray:
        pts2d = []
        for p in _FACE_3D:
            q = cam @ (R_mat @ p + t)
            pts2d.append((q[0] / q[2], q[1] / q[2]))
        return np.array(pts2d, dtype=np.float32)

    src_pts = _project(R)               # current pose projected onto image
    dst_pts = _project(np.eye(3))       # frontal pose projected onto image

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None or (mask is not None and mask.sum() < 3):
        return None, 0.0

    h, w = img_bgr.shape[:2]
    warped = cv2.warpPerspective(img_bgr, H, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    return warped, 0.62


class Frontalizer:
    """Frontalization pipeline.  Tries 3DDFA_V2 first, falls back to PnP warp.

    Usage:
        frontalizer = Frontalizer()
        frontalized_img, conf_mult = frontalizer.frontalize(img_bgr, bbox, kps_5pt)
        # Then re-detect on frontalized_img and apply standard 5-pt alignment.
    """

    def __init__(self, tddfa_root: Optional[str] = None):
        self._tddfa        = None
        self._backend      = "pnp_fallback"
        self._tddfa_root   = tddfa_root
        self._try_load_tddfa()

    # ------------------------------------------------------------------
    def _try_load_tddfa(self) -> None:
        """Attempt to import and initialise 3DDFA_V2."""
        try:
            import sys, yaml

            # Common locations where someone might have cloned 3DDFA_V2
            search_roots = [
                self._tddfa_root,
                os.path.expanduser("~/3DDFA_V2"),
                os.path.join(os.path.dirname(__file__), "..", "3DDFA_V2"),
                "3DDFA_V2",
            ]
            root = None
            for r in search_roots:
                if r and os.path.isdir(r):
                    root = os.path.abspath(r)
                    break
            if root is None:
                return

            if root not in sys.path:
                sys.path.insert(0, root)

            from TDDFA import TDDFA  # noqa: PLC0415  (optional import)

            cfg_path = os.path.join(root, "configs", "mb1_120x120.yml")
            if not os.path.exists(cfg_path):
                return

            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)

            self._tddfa    = TDDFA(gpu_mode=False, **cfg)
            self._backend  = "3ddfa_v2"
            print("[Frontalizer] 3DDFA_V2 loaded successfully.")

        except Exception as exc:
            print(f"[Frontalizer] 3DDFA_V2 not available ({exc}); using PnP fallback.")

    # ------------------------------------------------------------------
    @property
    def backend(self) -> str:
        return self._backend

    # ------------------------------------------------------------------
    def frontalize(
        self,
        img_bgr: np.ndarray,
        bbox: np.ndarray,
        kps_5pt: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Return (frontalized_bgr_image, confidence_multiplier).

        confidence_multiplier is in (0, 1]; lower means less reliable.
        Returns (None, 0.0) if frontalization completely fails.
        """
        if self._tddfa is not None:
            result = self._frontalize_tddfa(img_bgr, bbox, kps_5pt)
            if result[0] is not None:
                return result
            # fall through to PnP on 3DDFA_V2 failure
        return _pnp_frontalize(img_bgr, kps_5pt)

    # ------------------------------------------------------------------
    def _frontalize_tddfa(
        self,
        img_bgr: np.ndarray,
        bbox: np.ndarray,
        kps_5pt: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Use 3DDFA_V2 to generate a frontalized crop."""
        try:
            import sys, os as _os

            # 3DDFA_V2 frontalization utility is in their utils/ package
            root = next(p for p in sys.path if _os.path.isdir(_os.path.join(p, "utils")))
            import importlib.util as _ilu

            def _import(mod, path):
                spec = _ilu.spec_from_file_location(mod, path)
                m    = _ilu.module_from_spec(spec)
                spec.loader.exec_module(m)
                return m

            frontalize_mod = _import(
                "tddfa_frontalize",
                _os.path.join(root, "utils", "frontalize.py"),
            )

            box = [float(bbox[0]), float(bbox[1]),
                   float(bbox[2]), float(bbox[3]), 1.0]
            param_lst, roi_box_lst = self._tddfa(img_bgr, [box])
            ver_lst  = self._tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)

            # frontalize() returns a synthesized RGB image
            front_rgb = frontalize_mod.frontalize(
                img_bgr[:, :, ::-1],  # BGR -> RGB
                param_lst,
                roi_box_lst,
                self._tddfa.bfm,
                ver_lst,
            )
            if front_rgb is None:
                return None, 0.0
            return front_rgb[:, :, ::-1], 0.82   # RGB -> BGR, conf multiplier

        except Exception as exc:
            print(f"[Frontalizer] 3DDFA_V2 frontalize() failed: {exc}")
            return None, 0.0
