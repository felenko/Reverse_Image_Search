# FRWorkerWithAlignment

An improved face recognition indexing and search pipeline built on top of the base FRWorker.
Replaces Haar-cascade detection and simple crop-and-resize alignment with a yaw-aware
multi-stage pipeline that adapts its alignment strategy based on head pose.

---

## Models used

| Model | Source | Purpose |
|-------|--------|---------|
| `det_10g` | insightface buffalo_l | RetinaFace face detection, 5-pt landmarks |
| `2d106det` | insightface buffalo_l | 106-pt 2D landmarks |
| `1k3d68` | insightface buffalo_l | 3D landmarks → accurate yaw/pitch/roll |
| `w600k_r50` | insightface buffalo_l | ArcFace 512-d face embedding |
| 3DDFA_V2 | github/cleardusk _(optional)_ | 3D morphable model frontalization |

All insightface models are downloaded automatically (~500 MB) on first run to
`~/.insightface/models/buffalo_l/`.

---

## Alignment strategy

Yaw angle is estimated from insightface's `1k3d68` 3D landmark model (accurate to ±90°)
with a geometric 5-pt heuristic as fallback.

| \|yaw\| | Alignment method | Confidence multiplier |
|---------|-----------------|----------------------|
| < 20° | **Similarity transform** (4-DOF) from 5-pt keypoints — minimal distortion | 1.00 |
| 20 – 45° | **Full affine** (6-DOF) from 5-pt keypoints — corrects lateral foreshortening | 0.75 – 1.00 |
| > 45° | **Frontalize → re-detect → similarity 5-pt** | 0.82 (3DDFA_V2) / 0.62 (PnP fallback) |

All output crops are **112×112 BGR**, aligned to ArcFace canonical landmark positions.

`confidence` stored per face = `det_score × alignment_multiplier`.
Use `?min_conf=` at search time to filter out low-confidence indexed entries.

> **Note on image resolution:** ArcFace embeddings degrade significantly on images
> smaller than ~200×200px. With proper-resolution source images (300px+) expect
> cross-yaw same-person similarities of 0.85–0.98. Tiny crops (< 120px) will yield
> lower scores (~0.55–0.75) regardless of alignment quality.

---

## Installation

```bash
cd FRWorkerWithAlignment
pip install -r requirements.txt
```

For GPU inference replace `onnxruntime` with `onnxruntime-gpu` in `requirements.txt`
before installing, then pass `--device cuda`.

### Tight face crops (no background margin)

If your source images are already tightly cropped to the face (no surrounding context),
the pipeline automatically adds padding before detection. No configuration needed.

### Optional: 3DDFA_V2 (better frontalization for |yaw| > 45°)

Without 3DDFA_V2 the pipeline uses a PnP-homography fallback (still functional,
confidence multiplier 0.62). With 3DDFA_V2 full 3D morphable model frontalization
is used (confidence multiplier 0.82).

**Automated install (recommended):**
```bash
python setup_3ddfa.py
# installs to ~/3DDFA_V2 by default
# patches FaceBoxes NMS and Sim3DR with pure-Python fallbacks if no C compiler

python setup_3ddfa.py --target D:/libs/3DDFA_V2   # custom location
```

The script handles:
- Cloning the repo (shallow)
- Installing pip dependencies
- Building Cython extensions (Sim3DR for rendering, FaceBoxes NMS for detection)
- Patching NMS and Sim3DR with pure-Python fallbacks if a C compiler is unavailable
- Verifying the install

**Manual install (if automated fails):**
```bash
git clone --depth=1 https://github.com/cleardusk/3DDFA_V2.git ~/3DDFA_V2
cd ~/3DDFA_V2
pip install -r requirements.txt
python build_cython.py build_ext --inplace   # requires MSVC / gcc
```

On Windows without a C compiler, MSVC Build Tools can be installed free:
```
winget install Microsoft.VisualStudio.2022.BuildTools
```

---

## Indexing

Place face images under any folder (default: `templates/`), subfolders are scanned
recursively. The relative path becomes the filename stored in the database.

```bash
# Index templates/ folder on CPU (default)
python worker.py

# Custom folder
python worker.py /path/to/photos

# GPU + 3DDFA_V2 frontalization
python worker.py /path/to/photos --db face_db_aligned --device cuda --tddfa-root ~/3DDFA_V2

# Lower detection threshold (helps with non-photographic or synthetic images)
python worker.py templates/ --det-thresh 0.3

# Index multiple faces per image
python worker.py templates/ --max-faces 3
```

The worker prints per-image alignment decisions and a method breakdown summary:

```
   [  1/120]  id=84921  yaw= +3.2°  method=sim5pt        conf=0.971  person_a/photo1.jpg
   [  2/120]  id=11043  yaw=+28.7°  method=affine5pt     conf=0.831  person_b/photo2.jpg
   [  3/120]  id=55219  yaw=+51.1°  method=frontal_pnp   conf=0.618  person_c/photo3.jpg
   [skip]     no face detected      person_d/extreme_profile.jpg

────────────────────────────────────────────────────────
Done.  indexed=118  skipped=2  errors=0
Alignment method breakdown:
  sim5pt             82
  affine5pt          31
  frontal_pnp         5
```

Indexed data is stored in `face_db_aligned/` (LanceDB directory format, safe for
concurrent writers across multiple worker processes).

**Re-indexing:** delete `face_db_aligned/` and re-run the worker.

---

## Search web app

```bash
python app.py --db face_db_aligned --images-dir templates/

# With 3DDFA_V2
python app.py --db face_db_aligned --images-dir templates/ --tddfa-root ~/3DDFA_V2

# Custom port / host
python app.py --db face_db_aligned --images-dir templates/ --port 5001 --host 0.0.0.0
```

Open `http://127.0.0.1:5000` in a browser.

**UI features:**
- Drag-and-drop or paste from clipboard
- Sliders for result count (k) and minimum similarity threshold
- Per-query alignment info bar (yaw, method, det score, confidence)
- Per-match similarity bar + alignment method badge + confidence

### REST API

`POST /api/search`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | multipart file | — | Image upload |
| `image` | form/JSON field | — | Base64 data URL or raw base64 |
| `k` | query int | 5 | Max results to return |
| `min_sim` | query float | 0.50 | Minimum cosine similarity |
| `min_conf` | query float | 0.0 | Minimum indexed face confidence |

**Example:**
```bash
curl -F "file=@photo.jpg" "http://127.0.0.1:5000/api/search?k=10&min_sim=0.60"
```

**Response:**
```json
{
  "ok": true,
  "query": {
    "yaw": 12.3,
    "alignmentMethod": "sim5pt",
    "detScore": 0.97,
    "confidence": 0.97
  },
  "matches": [
    {
      "id": 611298077189,
      "imageFileName": "person_a/photo1.jpg",
      "similarity": 0.9841,
      "alignmentMethod": "sim5pt",
      "confidence": 0.971
    },
    {
      "id": 500858932096,
      "imageFileName": "person_a/photo2.jpg",
      "similarity": 0.8732,
      "alignmentMethod": "affine5pt",
      "confidence": 0.831
    }
  ]
}
```

Similarity thresholds (approximate, high-res images):

| Similarity | Interpretation |
|-----------|----------------|
| > 0.90 | Almost certainly same person |
| 0.75 – 0.90 | Likely same person, moderate yaw difference |
| 0.55 – 0.75 | Possible match, high yaw or low image quality |
| < 0.55 | Likely different person |

`GET /images/<path>` — serves indexed images from `--images-dir`.

---

## Project layout

```
FRWorkerWithAlignment/
├── face_detector.py      RetinaFace + 106-pt + 3D landmarks + pose (insightface)
├── face_aligner.py       Yaw-based alignment routing (sim5pt / affine5pt / frontal)
├── frontalizer.py        3DDFA_V2 or PnP-homography fallback frontalization
├── embedding_model.py    ArcFace extractor (insightface w600k_r50)
├── pipeline.py           FacePipeline — auto-padding, detection, alignment, embedding
├── worker.py             Indexing CLI
├── app.py                Flask search app + REST API
├── database.py           LanceDB helpers (schema: vector, yaw, method, confidence)
├── setup_3ddfa.py        Automated 3DDFA_V2 installer with pure-Python fallback patches
├── requirements.txt
└── web_templates/
    └── index.html        Search UI
```

---

## Comparison with base FRWorker

| | FRWorker | FRWorkerWithAlignment |
|--|----------|----------------------|
| Detection | OpenCV Haar cascade | RetinaFace (neural, much more accurate) |
| Alignment | Simple crop + resize | Yaw-adaptive (sim5pt / affine / frontalization) |
| Pose estimation | None | 3D landmarks → accurate yaw/pitch/roll |
| Embedding | AdaFace IR50 | ArcFace w600k_r50 |
| DB schema | vector, filename | + yaw, alignment_method, confidence |
| Large yaw handling | Poor (distorted crop) | Frontalization (PnP or 3DDFA_V2) |
| Tight crop handling | Works | Auto-padded before detection |
