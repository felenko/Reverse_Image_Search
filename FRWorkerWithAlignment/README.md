# FRWorkerWithAlignment

Improved face recognition worker using:
- **RetinaFace** (via insightface `det_10g`) — accurate face detection + 5-pt landmarks
- **2d106det** (insightface) — 106-point 2D landmarks + pose estimation (yaw/pitch/roll)
- **ArcFace** (insightface `w600k_r50`) — 512-d face embeddings
- **3DDFA_V2** _(optional)_ — full 3D morphable model frontalization for large yaw

## Alignment strategy

| |yaw| range | Method | Confidence multiplier |
|-------------|--------|----------------------|
| < 20° | Similarity transform (4-DOF) from 5-pt keypoints | 1.00 |
| 20 – 45° | Full affine (6-DOF) from 5-pt keypoints, compensates foreshortening | 0.75 – 1.00 |
| > 45° | Frontalize → re-detect → similarity 5-pt | 0.62 (PnP) / 0.82 (3DDFA_V2) |

All crops are 112×112 BGR, normalized to ArcFace canonical pose.
The `confidence` stored per face = `det_score × alignment_multiplier`.

---

## Installation

```bash
cd FRWorkerWithAlignment
pip install -r requirements.txt
```

> For CUDA inference replace `onnxruntime` with `onnxruntime-gpu`.

### Optional: 3DDFA_V2 (3D frontalization for |yaw| > 45°)

Without 3DDFA_V2, large-yaw faces use the PnP-homography fallback (confidence 0.62).
With 3DDFA_V2 they are fully frontalized using a 3D morphable model (confidence 0.82).

```bash
# Clone and build 3DDFA_V2 (requires a C compiler)
git clone https://github.com/cleardusk/3DDFA_V2.git ~/3DDFA_V2
cd ~/3DDFA_V2
pip install -r requirements.txt
python build_cython.py build_ext --inplace

# Download BFM model weights
python -c "from utils.functions import *"  # triggers auto-download
```

Then pass `--tddfa-root ~/3DDFA_V2` to `worker.py` / `app.py`.

---

## Indexing

```bash
# Index the templates/ folder (CPU)
python worker.py

# Index a custom folder using CUDA and 3DDFA_V2
python worker.py /path/to/photos --db face_db_aligned --device cuda --tddfa-root ~/3DDFA_V2

# Show help
python worker.py --help
```

Indexed data is stored in `face_db_aligned/` (LanceDB format, safe for concurrent writers).

## Search web app

```bash
python app.py --db face_db_aligned --images-dir templates/
# open http://127.0.0.1:5000
```

The `/api/search` endpoint accepts `POST` with `file` (multipart) or `image` (base64).
Response includes per-query alignment metadata:
```json
{
  "ok": true,
  "query": { "yaw": 32.1, "alignmentMethod": "affine5pt", "detScore": 0.97, "confidence": 0.88 },
  "matches": [
    { "id": 123, "imageFileName": "person/photo.jpg", "similarity": 0.9812,
      "alignmentMethod": "sim5pt", "confidence": 0.96 }
  ]
}
```

---

## Project layout

```
FRWorkerWithAlignment/
├── face_detector.py    RetinaFace + 106-pt landmarks + pose (insightface)
├── face_aligner.py     Yaw-based alignment routing (sim5pt / affine5pt / frontal)
├── frontalizer.py      3DDFA_V2 or PnP-homography frontalization
├── embedding_model.py  ArcFace extractor (insightface w600k_r50)
├── pipeline.py         FacePipeline (detector + aligner + embedder)
├── worker.py           Indexing CLI
├── app.py              Flask search app
├── database.py         LanceDB helpers (extended schema)
└── requirements.txt
```
