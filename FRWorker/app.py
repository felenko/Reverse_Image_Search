"""
Simple web app: upload or paste image -> compute AdaFace embedding -> lookup in DB -> show matched file.
"""
import argparse
import base64
import io
from pathlib import Path

from PIL import Image
from flask import Flask, request, jsonify, send_file, render_template

from adaface_model import AdaFaceExtractor
from database import open_table, find_nearest

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# Set by main()
EXTRACTOR: AdaFaceExtractor = None
DB_TABLE = None
IMAGES_DIR: Path = None


def get_embedding_from_request():
    """Get image from request: either 'file' (upload) or 'image' (base64 data URL / raw base64)."""
    if "file" in request.files:
        f = request.files["file"]
        if f.filename:
            data = f.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
            return EXTRACTOR.get_embedding_from_image(img)

    if "image" in request.form:
        raw = request.form["image"].strip()
        if raw.startswith("data:image"):
            raw = raw.split(",", 1)[-1]
        try:
            data = base64.b64decode(raw)
        except Exception:
            return None
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return EXTRACTOR.get_embedding_from_image(img)

    if request.is_json and "image" in request.json:
        raw = request.json["image"].strip()
        if raw.startswith("data:image"):
            raw = raw.split(",", 1)[-1]
        try:
            data = base64.b64decode(raw)
        except Exception:
            return None
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return EXTRACTOR.get_embedding_from_image(img)

    return None


@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    tb = traceback.format_exc()
    print("UNHANDLED EXCEPTION:\n", tb)
    return jsonify({"ok": False, "error": tb}), 500

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search", methods=["POST"])
def search():
    import traceback
    try:
        emb = get_embedding_from_request()
        if emb is None:
            return jsonify({"ok": False, "error": "No image provided or could not decode (use 'file' or 'image' base64)"}), 400
        MIN_SIMILARITY = 0.95
        top = find_nearest(DB_TABLE, emb, top_k=request.args.get("k", 5, type=int))
        results = [
            {"id": id_, "imageFileName": name, "similarity": round(sim, 4)}
            for id_, name, sim in top
            if sim >= MIN_SIMILARITY
        ]
        return jsonify({"ok": True, "matches": results})
    except Exception:
        err = traceback.format_exc()
        print("SEARCH ERROR:\n", err)
        return jsonify({"ok": False, "error": err}), 500


@app.route("/images/<path:filename>")
def serve_image(filename):
    """Serve an image from the indexed images folder."""
    if IMAGES_DIR is None:
        return "Images directory not configured", 404
    if ".." in filename:
        return "Forbidden", 403
    filename = filename.replace("\\", "/").strip("/")
    parts = [p for p in filename.split("/") if p]
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
    return send_file(path, mimetype=None, as_attachment=False)


def main():
    global EXTRACTOR, DB_TABLE, IMAGES_DIR
    _app_dir = Path(__file__).resolve().parent
    _default_images_dir = (_app_dir / "templates").resolve()
    parser = argparse.ArgumentParser(description="Face lookup web app")
    parser.add_argument("--db", type=str, default="face_db", help="LanceDB directory path (default: face_db)")
    parser.add_argument("--images-dir", type=str, default=str(_default_images_dir),
                        help="Root folder for images (default: templates/)")
    parser.add_argument("--cache-dir", type=str, default=None, help="AdaFace model cache directory")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    DB_TABLE = open_table(args.db)
    print(f"Database: {args.db}  ({len(DB_TABLE)} faces indexed)")

    IMAGES_DIR = Path(args.images_dir).resolve() if args.images_dir else None
    print(f"Images root: {IMAGES_DIR}")
    if IMAGES_DIR and not IMAGES_DIR.is_dir():
        print(f"Warning: images-dir does not exist: {IMAGES_DIR}")

    EXTRACTOR = AdaFaceExtractor(cache_dir=args.cache_dir)
    # Pre-load the model now (triggers os.chdir internally) before any requests arrive
    print("Pre-loading AdaFace model...")
    _ = EXTRACTOR.model
    print("Model ready.")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
