"""
Search web app for FRWorkerWithAlignment.

Upload an image -> RetinaFace + yaw-based alignment + ArcFace -> nearest matches.
Inherits the same Flask structure as the original FRWorker app.

Start:
    python app.py [--db face_db_aligned] [--images-dir templates/] [--port 5000]
"""
import argparse
import base64
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from flask import Flask, jsonify, render_template, request, send_file

from database import find_nearest, open_table
from pipeline import FacePipeline

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# Globals set by main()
PIPELINE:    FacePipeline = None
DB_TABLE                  = None
IMAGES_DIR:  Path         = None


# ---------------------------------------------------------------------------
# Image decode helpers
# ---------------------------------------------------------------------------

def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image bytes")
    return img_bgr


def _get_image_bgr_from_request() -> np.ndarray | None:
    if "file" in request.files:
        f = request.files["file"]
        if f.filename:
            return _decode_image(f.read())

    for src in (request.form, request.json or {}):
        if "image" in src:
            raw = src["image"].strip()
            if raw.startswith("data:image"):
                raw = raw.split(",", 1)[-1]
            try:
                return _decode_image(base64.b64decode(raw))
            except Exception:
                return None

    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    return jsonify({"ok": False, "error": traceback.format_exc()}), 500


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search", methods=["POST"])
def search():
    import traceback
    try:
        img_bgr = _get_image_bgr_from_request()
        if img_bgr is None:
            return jsonify({"ok": False, "error": "No image provided"}), 400

        top_k          = request.args.get("k", 5, type=int)
        min_confidence = request.args.get("min_conf", 0.0, type=float)
        min_similarity = request.args.get("min_sim", 0.92, type=float)

        results_pipeline = PIPELINE.process_bgr(img_bgr, max_faces=1)
        if not results_pipeline:
            return jsonify({"ok": True, "matches": [], "detail": "No face detected in query image"})

        face = results_pipeline[0]
        if face.embedding is None:
            return jsonify({"ok": True, "matches": [], "detail": "Embedding extraction failed"})

        db_matches = find_nearest(
            DB_TABLE,
            face.embedding,
            top_k=top_k,
            min_confidence=min_confidence,
        )

        matches = [
            {
                "id":              id_,
                "imageFileName":   name,
                "similarity":      sim,
                "alignmentMethod": method,
                "confidence":      conf,
            }
            for id_, name, sim, method, conf in db_matches
            if sim >= min_similarity
        ]

        return jsonify({
            "ok":      True,
            "matches": matches,
            "query": {
                "yaw":             round(face.yaw, 1),
                "alignmentMethod": face.alignment_method,
                "detScore":        face.det_score,
                "confidence":      face.confidence,
            },
        })

    except Exception:
        err = traceback.format_exc()
        print("SEARCH ERROR:\n", err)
        return jsonify({"ok": False, "error": err}), 500


@app.route("/images/<path:filename>")
def serve_image(filename: str):
    if IMAGES_DIR is None:
        return "Images directory not configured", 404
    if ".." in filename:
        return "Forbidden", 403
    filename = filename.replace("\\", "/").strip("/")
    parts    = [p for p in filename.split("/") if p]
    if not parts:
        return "Not found", 404
    base = IMAGES_DIR.resolve()
    path = base.joinpath(*parts).resolve()
    try:
        path.relative_to(base)
    except ValueError:
        return "Forbidden", 403
    if not path.is_file():
        return f"Not found: {path}", 404
    return send_file(path, as_attachment=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global PIPELINE, DB_TABLE, IMAGES_DIR

    _here          = Path(__file__).resolve().parent
    _default_imgs  = _here / "templates"

    parser = argparse.ArgumentParser(description="Alignment-aware face search web app")
    parser.add_argument("--db",          default="face_db_aligned")
    parser.add_argument("--images-dir",  default=str(_default_imgs), dest="images_dir")
    parser.add_argument("--device",      default=None, choices=["cpu", "cuda"])
    parser.add_argument("--tddfa-root",  default=None, dest="tddfa_root")
    parser.add_argument("--port",        type=int, default=5000)
    parser.add_argument("--host",        default="127.0.0.1")
    args = parser.parse_args()

    DB_TABLE   = open_table(args.db)
    IMAGES_DIR = Path(args.images_dir).resolve() if args.images_dir else None

    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    PIPELINE = FacePipeline(device=device, tddfa_root=args.tddfa_root)

    print(f"Database    : {args.db}  ({len(DB_TABLE)} faces indexed)")
    print(f"Images root : {IMAGES_DIR}")
    print(f"Device      : {device}")
    print("Pre-loading models…")
    _ = PIPELINE.detector
    _ = PIPELINE.aligner
    _ = PIPELINE.embedder
    print(f"Ready.  Frontalizer: {PIPELINE.aligner._frontalizer.backend}")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
