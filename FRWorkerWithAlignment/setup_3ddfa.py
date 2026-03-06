"""
Automated installer for 3DDFA_V2.

Run once before using the frontalizer with --tddfa-root:

    python setup_3ddfa.py
    python setup_3ddfa.py --target ~/my_dir   # install elsewhere

What this does:
  1. Clones https://github.com/cleardusk/3DDFA_V2 (shallow)
  2. Installs its pip requirements into the current environment
  3. Tries to build Cython extensions (Sim3DR, FaceBoxes NMS, face3d mesh)
     - If the C compiler is missing, patches BOTH FaceBoxes NMS (torchvision)
       AND Sim3DR_Cython (OpenCV painter's algorithm) with pure-Python fallbacks
       so everything works without MSVC/MinGW.
  4. Verifies pretrained model weights are present
  5. Prints the --tddfa-root path to use in worker.py / app.py

NOTE for FRWorkerWithAlignment:
  FaceBoxes is NOT used by our pipeline (we supply RetinaFace bboxes directly
  to TDDFA).  Only Sim3DR is required for frontalization rendering.
  The NMS patch is applied so `demo.py` also works without a C compiler.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path


REPO_URL     = "https://github.com/cleardusk/3DDFA_V2.git"
DEFAULT_ROOT = Path.home() / "3DDFA_V2"

REQUIRED_WEIGHTS = [
    "weights/mb1_120x120.pth",
    "weights/mb05_120x120.pth",
]

BUILD_SCRIPT = "build_cython.py"

# Pure-Python NMS replacement injected into FaceBoxes/utils/nms_wrapper.py
# when the Cython cpu_nms extension cannot be compiled.
# Uses torchvision.ops.nms (always available when torch+torchvision are installed).
_NMS_PATCH = textwrap.dedent("""\
    # ---- patched by setup_3ddfa.py: pure-Python NMS (no Cython required) ----
    import numpy as np
    import torch
    from torchvision.ops import nms as _tv_nms


    def cpu_nms(dets: np.ndarray, thresh: float) -> np.ndarray:
        \"\"\"Drop-in replacement for the Cython cpu_nms using torchvision.\"\"\"
        if dets.shape[0] == 0:
            return np.empty((0,), dtype=np.int32)
        boxes  = torch.as_tensor(dets[:, :4], dtype=torch.float32)
        scores = torch.as_tensor(dets[:,  4], dtype=torch.float32)
        keep   = _tv_nms(boxes, scores, float(thresh))
        return keep.numpy().astype(np.int32)


    def cpu_soft_nms(
        dets: np.ndarray,
        sigma: float = 0.5,
        Nt: float = 0.3,
        threshold: float = 0.001,
        method: int = 1,
    ) -> np.ndarray:
        \"\"\"Soft-NMS via score decay (pure Python fallback).\"\"\"
        if dets.shape[0] == 0:
            return dets
        dets = dets.copy()
        x1, y1, x2, y2, scores = (dets[:, i] for i in range(5))
        areas  = (x2 - x1 + 1) * (y2 - y1 + 1)
        for i in range(dets.shape[0]):
            max_idx = scores[i:].argmax() + i
            dets[[i, max_idx]] = dets[[max_idx, i]]
            areas[[i, max_idx]] = areas[[max_idx, i]]
            ix1 = np.maximum(x1[i], x1[i + 1:])
            iy1 = np.maximum(y1[i], y1[i + 1:])
            ix2 = np.minimum(x2[i], x2[i + 1:])
            iy2 = np.minimum(y2[i], y2[i + 1:])
            inter = np.maximum(0.0, ix2 - ix1 + 1) * np.maximum(0.0, iy2 - iy1 + 1)
            ovr   = inter / (areas[i] + areas[i + 1:] - inter)
            if method == 1:      # linear
                weight = np.where(ovr > Nt, 1.0 - ovr, 1.0)
            elif method == 2:    # gaussian
                weight = np.exp(-(ovr * ovr) / sigma)
            else:                # hard NMS
                weight = np.where(ovr > Nt, 0.0, 1.0)
            scores[i + 1:] *= weight
        return dets[scores > threshold]


# Alias expected by FaceBoxes/FaceBoxes.py: `from .utils.nms_wrapper import nms`
nms = cpu_nms
# ---- end patch ----
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> int:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if check and result.returncode != 0:
        print(f"\n[ERROR] Command failed (exit {result.returncode}).", file=sys.stderr)
        sys.exit(result.returncode)
    return result.returncode


def pip_install(packages: list[str]) -> None:
    run([sys.executable, "-m", "pip", "install", "--upgrade", *packages])


def git_available() -> bool:
    return shutil.which("git") is not None


# ---------------------------------------------------------------------------
# Step 1 – Clone
# ---------------------------------------------------------------------------

def step_clone(target: Path) -> None:
    print(f"\n[1/5] Cloning 3DDFA_V2 -> {target}")
    if target.exists():
        if (target / ".git").is_dir():
            print("  Repository already exists; pulling latest changes.")
            run(["git", "pull", "--ff-only"], cwd=target, check=False)
            return
        print(f"  Directory exists but is not a git repo: {target}")
        print("  Delete it manually or choose a different --target.")
        sys.exit(1)

    target.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth=1", REPO_URL, str(target)])


# ---------------------------------------------------------------------------
# Step 2 – pip requirements
# ---------------------------------------------------------------------------

def step_pip_requirements(target: Path) -> None:
    print("\n[2/5] Installing pip requirements")
    req = target / "requirements.txt"
    if req.exists():
        pip_install(["-r", str(req)])
    else:
        print("  requirements.txt not found; installing known deps manually.")
        pip_install([
            "torch", "torchvision",
            "opencv-python", "numpy", "scipy",
            "scikit-image", "matplotlib", "tqdm",
            "pyyaml", "onnxruntime", "Cython",
        ])


# ---------------------------------------------------------------------------
# Step 3 – Build Cython (with per-module fallback)
# ---------------------------------------------------------------------------

def _try_build(target: Path, setup_arg: str, label: str) -> bool:
    """Try one Cython build; return True on success."""
    rc = run(
        [sys.executable, BUILD_SCRIPT, "build_ext", "--inplace", "--", setup_arg],
        cwd=target, check=False,
    )
    # build_cython.py doesn't accept extra args in older versions; retry plain
    if rc != 0:
        rc = run(
            [sys.executable, BUILD_SCRIPT, "build_ext", "--inplace"],
            cwd=target, check=False,
        )
    if rc == 0:
        exts = list(target.rglob("*.pyd")) + list(target.rglob("*.so"))
        print(f"  {label}: built ({len(exts)} extension file(s) total)")
        return True
    print(f"  {label}: build FAILED (exit {rc}) – will apply fallback patch.")
    return False


def step_build_cython(target: Path) -> dict[str, bool]:
    """
    Build Cython extensions.  Returns dict of {module: success}.
    The only extension critical for FRWorkerWithAlignment is Sim3DR.
    FaceBoxes NMS is needed only for demo.py; we patch it if build fails.
    """
    print("\n[3/5] Building Cython extensions")
    build_script = target / BUILD_SCRIPT
    if not build_script.exists():
        print(f"  {BUILD_SCRIPT} not found – skipping Cython build entirely.")
        return {}

    rc = run(
        [sys.executable, BUILD_SCRIPT, "build_ext", "--inplace"],
        cwd=target, check=False,
    )
    built_files = list(target.rglob("*.pyd")) + list(target.rglob("*.so"))

    results = {
        "Sim3DR":    any("Sim3DR"    in str(f) for f in built_files),
        "cpu_nms":   any("cpu_nms"   in str(f) for f in built_files),
        "face3d":    any("mesh_core" in str(f) for f in built_files),
    }

    for mod, ok in results.items():
        status = "OK" if ok else "MISSING"
        print(f"    {mod:<12} {status}")

    if not results.get("Sim3DR"):
        print(
            "\n  WARNING: Sim3DR not built.\n"
            "  Frontalization rendering in FRWorkerWithAlignment will be unavailable.\n"
            "  Install a C compiler (MSVC / MinGW on Windows, gcc on Linux) and re-run."
        )

    return results


# ---------------------------------------------------------------------------
# Step 4 – Patch FaceBoxes NMS if Cython build failed
# ---------------------------------------------------------------------------

def step_patch_nms(target: Path, build_results: dict[str, bool]) -> None:
    """
    If cpu_nms.pyd/.so was not built, replace nms_wrapper.py with a pure-Python
    implementation backed by torchvision.ops.nms.

    This lets demo.py and any FaceBoxes-dependent code work without a C compiler.
    FRWorkerWithAlignment does NOT use FaceBoxes (it passes RetinaFace bboxes
    directly to TDDFA), so this patch is only needed for the 3DDFA_V2 demo.
    """
    if build_results.get("cpu_nms"):
        print("\n[4/5] NMS patch: not needed (cpu_nms compiled successfully).")
        return

    print("\n[4/5] NMS patch: cpu_nms not compiled – injecting torchvision fallback.")

    wrapper = target / "FaceBoxes" / "utils" / "nms_wrapper.py"
    if not wrapper.exists():
        print(f"  {wrapper} not found – skipping patch.")
        return

    original = wrapper.read_text(encoding="utf-8")

    # Already patched?
    if "patched by setup_3ddfa.py" in original:
        print("  Already patched.")
        return

    # Back up the original
    backup = wrapper.with_suffix(".py.orig")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
        print(f"  Backed up original -> {backup.name}")

    wrapper.write_text(_NMS_PATCH, encoding="utf-8")
    print(f"  Patched {wrapper.relative_to(target)}")

    # Verify the patch imports cleanly
    check = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0,{str(target)!r}); "
         "from FaceBoxes.utils.nms_wrapper import cpu_nms; print('NMS OK')"],
        capture_output=True, text=True,
    )
    if "NMS OK" in check.stdout:
        print("  Patch verified successfully.")
    else:
        print(f"  Patch verification failed:\n{check.stderr.strip()}")
        print("  (torchvision may not be installed – run: pip install torchvision)")


# ---------------------------------------------------------------------------
# Step 4b – Patch Sim3DR_Cython if Cython build failed
# ---------------------------------------------------------------------------

_SIM3DR_PATCH = '''\
# ---- patched by setup_3ddfa.py: pure-Python Sim3DR_Cython (no Cython required) ----
#
# Exact in-place calling convention from Sim3DR/Sim3DR.py:
#   get_normal(normal_out, vertices, triangles, nver, ntri)
#   rasterize(bg, vertices, triangles, colors, z_buffer,
#             ntri, height, width, channel, reverse=False)
import cv2
import numpy as np


def get_normal(normal, vertices, triangles, nver, ntri):
    """Accumulate per-triangle normals into vertex normals (in-place)."""
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    tri_n = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    normal[:] = 0.0
    np.add.at(normal, triangles[:, 0], tri_n)
    np.add.at(normal, triangles[:, 1], tri_n)
    np.add.at(normal, triangles[:, 2], tri_n)
    mag = np.linalg.norm(normal, axis=1, keepdims=True)
    normal /= np.maximum(mag, 1e-8)


def rasterize(bg, vertices, triangles, colors, buffer,
              ntri, height, width, channel, reverse=False):
    """Painter\'s-algorithm rasterizer using OpenCV fillConvexPoly (in-place)."""
    pts_all = vertices[triangles, :2]
    z_mean  = vertices[triangles, 2].mean(axis=1)

    x_max = pts_all[:, :, 0].max(axis=1)
    x_min = pts_all[:, :, 0].min(axis=1)
    y_max = pts_all[:, :, 1].max(axis=1)
    y_min = pts_all[:, :, 1].min(axis=1)
    vis = (x_max >= 0) & (x_min < width) & (y_max >= 0) & (y_min < height)

    pts_v  = pts_all[vis]
    z_v    = z_mean[vis]
    tris_v = triangles[vis]

    order       = np.argsort(z_v) if not reverse else np.argsort(-z_v)
    pts_sorted  = pts_v[order].astype(np.int32)
    cols_sorted = colors[tris_v[order]].mean(axis=1)
    cols_sorted = np.clip(cols_sorted, 0, 255).astype(np.uint8)

    for i in range(len(order)):
        cv2.fillConvexPoly(bg, pts_sorted[i].reshape(-1, 1, 2),
                           cols_sorted[i, :channel].tolist())
# ---- end patch ----
'''


def step_patch_sim3dr(target: Path, build_results: dict[str, bool]) -> None:
    """Write Sim3DR_Cython.py fallback into Sim3DR/ if the .pyd was not built.

    _init_paths.py adds Sim3DR/ to sys.path, so placing the .py file there
    makes `import Sim3DR_Cython` succeed without any .pyd file.
    """
    if build_results.get("Sim3DR"):
        print("\n[4b] Sim3DR patch: not needed (Sim3DR_Cython compiled successfully).")
        return

    print("\n[4b] Sim3DR patch: Sim3DR_Cython not compiled – injecting pure-Python fallback.")

    dest = target / "Sim3DR" / "Sim3DR_Cython.py"
    if dest.exists() and "patched by setup_3ddfa.py" in dest.read_text(encoding="utf-8"):
        print("  Already patched.")
        return

    dest.write_text(_SIM3DR_PATCH, encoding="utf-8")
    print(f"  Written: {dest.relative_to(target)}")

    # Verify
    check = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {str(target / 'Sim3DR')!r}); "
         "import Sim3DR_Cython; print('Sim3DR OK')"],
        capture_output=True, text=True,
    )
    if "Sim3DR OK" in check.stdout:
        print("  Patch verified successfully.")
    else:
        print(f"  Patch verification failed:\n{check.stderr.strip()}")


# ---------------------------------------------------------------------------
# Step 5 – Verify weights
# ---------------------------------------------------------------------------

def step_verify_weights(target: Path) -> None:
    print("\n[5/5] Verifying model weights")
    missing = [w for w in REQUIRED_WEIGHTS if not (target / w).exists()]

    if not missing:
        print("  All required weights present.")
        return

    print(f"  Missing: {missing}")
    print("  Attempting to download …")

    # script.sh (Linux/Mac)
    dl_script = target / "script.sh"
    if dl_script.exists() and platform.system() != "Windows":
        if run(["bash", str(dl_script)], cwd=target, check=False) == 0:
            return

    # git-lfs
    if shutil.which("git-lfs"):
        if run(["git", "lfs", "pull"], cwd=target, check=False) == 0:
            return

    print(
        "\n  [ACTION REQUIRED] Could not auto-download weights.\n"
        "  Download .pth files from:\n"
        "    https://github.com/cleardusk/3DDFA_V2/releases\n"
        f"  and place them in: {target / 'weights'}\n"
    )


# ---------------------------------------------------------------------------
# Final verification
# ---------------------------------------------------------------------------

def verify_import(target: Path) -> bool:
    print("\nVerifying TDDFA import …")
    code = (
        f"import sys; sys.path.insert(0, {str(target)!r}); "
        "from TDDFA import TDDFA; print('TDDFA OK')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode == 0 and "TDDFA OK" in r.stdout:
        print("  TDDFA import: OK")
    else:
        print(f"  TDDFA import FAILED:\n{r.stderr.strip()}")
        return False

    # Verify Sim3DR (needed by frontalization)
    code_sim = (
        f"import sys; sys.path.insert(0,{str(target / 'Sim3DR')!r}); "
        "import Sim3DR_Cython; print('Sim3DR OK')"
    )
    r_sim = subprocess.run([sys.executable, "-c", code_sim], capture_output=True, text=True)
    if r_sim.returncode == 0 and "Sim3DR OK" in r_sim.stdout:
        print("  Sim3DR import: OK  (frontalization will work)")
    else:
        print(f"  Sim3DR import FAILED: {r_sim.stderr.strip().splitlines()[-1] if r_sim.stderr.strip() else '?'}")

    # Verify FaceBoxes (for demo.py)
    code2 = (
        f"import sys; sys.path.insert(0, {str(target)!r}); "
        "from FaceBoxes import FaceBoxes; print('FaceBoxes OK')"
    )
    r2 = subprocess.run([sys.executable, "-c", code2], capture_output=True, text=True)
    if r2.returncode == 0 and "FaceBoxes OK" in r2.stdout:
        print("  FaceBoxes import: OK  (demo.py will work)")
    else:
        print(f"  FaceBoxes import FAILED: {r2.stderr.strip().splitlines()[-1] if r2.stderr.strip() else '?'}")
        print("  demo.py may not work, but FRWorkerWithAlignment is unaffected.")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install 3DDFA_V2 for FRWorkerWithAlignment."
    )
    parser.add_argument(
        "--target", default=str(DEFAULT_ROOT),
        help=f"Where to clone 3DDFA_V2 (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--skip-build", action="store_true",
        help="Skip Cython build (applies NMS patch directly)",
    )
    args = parser.parse_args()
    target = Path(args.target).expanduser().resolve()

    print("=" * 60)
    print("  3DDFA_V2 automated installer")
    print(f"  Target : {target}")
    print(f"  Python : {sys.executable}")
    print(f"  OS     : {platform.system()} {platform.machine()}")
    print("=" * 60)

    if not git_available():
        print("[ERROR] 'git' not found in PATH.", file=sys.stderr)
        sys.exit(1)

    step_clone(target)
    step_pip_requirements(target)
    build_results = {} if args.skip_build else step_build_cython(target)
    step_patch_nms(target, build_results)
    step_patch_sim3dr(target, build_results)
    step_verify_weights(target)
    ok = verify_import(target)

    print("\n" + "=" * 60)
    if ok:
        print("  3DDFA_V2 installation complete.")
        print(f"\n  Run the demo:")
        print(f"    cd {target}")
        print(f"    python demo.py -f examples/inputs/trump_hillary.jpg")
        print(f"\n  Use with FRWorkerWithAlignment:")
        print(f"    python worker.py templates/ --tddfa-root {target}")
        print(f"    python app.py              --tddfa-root {target}")
    else:
        print("  Installation finished but TDDFA import failed.")
        print("  Sim3DR Cython build likely needs a C compiler.")
        if platform.system() == "Windows":
            print("  Install: winget install Microsoft.VisualStudio.2022.BuildTools")
        print(f"  Target path: {target}")
    print("=" * 60)


if __name__ == "__main__":
    main()
