# Face Recognition Worker & Lookup Web App

Uses **AdaFace** (CVLFace IR50 from Hugging Face) to compute face embeddings, stores them in SQLite, and provides a simple web UI to upload/paste an image and see the matching file from the indexed set.

## Setup

```bash
cd FRWorker
pip install -r requirements.txt
```

The first run will download the AdaFace model from Hugging Face into `~/.cvlface_cache/adaface_ir50_ms1mv2` (or the path you pass as `--cache-dir`).

## 1. Index face images (worker)

Put face images in a folder, then run the worker to compute embeddings and fill the database:

```bash
python worker.py path/to/input_folder [--db face_embeddings.db] [--device cuda]
```

- **input_folder**: folder containing face images (e.g. `.jpg`, `.png`).
- **--db**: SQLite file (default: `face_embeddings.db`).
- **--cache-dir**: directory for AdaFace model cache (optional).
- **--device**: `cpu` or `cuda` (default: auto).

Database schema:

- **Metadata**: `id` (PK), `imageFileName`
- **Embeddings**: `id` (PK, FK to Metadata), `embedding` (BLOB)

## 2. Run the web app

```bash
python app.py [--db face_embeddings.db] [--images-dir path/to/input_folder] [--port 5000]
```

- **--images-dir**: folder that contains the indexed images (same as worker input). Needed so the app can serve the matched image; if omitted, matches show filename only and the image URL may 404.
- **--db**: same SQLite file you used for the worker.

Open `http://127.0.0.1:5000/`. You can:

- **Upload** an image (file picker or drag-and-drop).
- **Paste** an image from the clipboard (Ctrl+V).

Click **Search** to compute the query embedding and find the nearest faces in the DB. The page shows the matched **imageFileName** and, if `--images-dir` is set, the image from the indexed pile.

## Summary

| Step        | Command |
|------------|--------|
| Index faces | `python worker.py path/to/faces --db face_embeddings.db` |
| Run web app | `python app.py --db face_embeddings.db --images-dir path/to/faces` |
