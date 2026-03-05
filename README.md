# Reverse Image Search

A face recognition and web crawling system for finding and indexing face images.

---

## Components

### `big-web-simulation/`

Simulates a larger web system hosting **Minimum Web** — three static portrait gallery sites, each with **100 dummy portraits** (men and women) loaded from `randomuser.me`.

| File | Purpose |
|---|---|
| `index.html` | Landing page linking to all three sites |
| `site1/`, `site2/`, `site3/` | Portrait gallery pages (100 portraits each) |
| `portraits.js` | Fetches portraits from `randomuser.me` API using a fixed seed per site |
| `styles.css` | Shared styles |

Portraits are seeded (`site1`, `site2`, `site3`) so the same 300 people appear on every run.

**To serve locally:**
```bash
cd big-web-simulation
python -m http.server 3000
# open http://localhost:3000/
```

---

### `FRWorker/`

Face recognition indexer and search API.

#### Tools

| Script | Purpose |
|---|---|
| `download_portraits.py` | Downloads all 300 portraits from the simulation into `templates/photos/` |
| `worker.py` | Scans `templates/` recursively, computes face embeddings, stores in SQLite |
| `app.py` | Flask web app — upload/paste an image, find the closest match in the DB |

#### Face Recognition — AdaFace (IR-50)

`worker.py` and `app.py` use **AdaFace** (`minchul/cvlface_adaface_ir50_ms1mv2`) for face embedding:

- Model: **IR-50** (IResNet-50) trained on MS1MV2, downloaded from Hugging Face on first run and cached at `~/.cvlface_cache/adaface_ir50_ms1mv2/`
- Embeddings: **512-dimensional** L2-normalised vectors stored in SQLite
- Similarity: **cosine similarity** used for nearest-neighbour lookup
- Face detection (pre-crop): **OpenCV Haar Cascade** (`haarcascade_frontalface_default.xml`) — fast CPU-only detection, crops and resizes face to 112×112 before passing to AdaFace

**Setup:**
```bash
cd FRWorker
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Index portraits:**
```bash
python worker.py                        # indexes templates/ recursively
python worker.py path/to/images         # or a custom folder
```

**Start search app:**
```bash
python app.py --images-dir templates/photos
```

---

### `WebCrawler/`

JS-aware web crawler that finds face images across crawled pages.

- Renders pages with a **headless Chromium** browser via **Playwright** — handles JavaScript-rendered content (dynamic image loading, SPAs, etc.)
- Extracts all links and image URLs from the fully-rendered DOM
- Downloads each image and runs **OpenCV Haar Cascade** face detection
- Saves images containing faces to an output folder
- Follows all links regardless of domain (configurable via `--max-pages`)

**Setup:**
```bash
cd WebCrawler
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
```

**Run:**
```bash
python crawler.py seeds_test.txt --output crawled_faces --verbose
```

| Flag | Default | Description |
|---|---|---|
| `--output` | `crawled_faces` | Folder to save face images |
| `--max-pages` | 0 (unlimited) | Stop after N pages |
| `--delay` | 0.5s | Pause between page loads |
| `--wait` | 2000ms | Extra wait after page load for JS to settle |
| `--verbose` | off | Show every image URL and detection result |

**Test with the simulation** — serve `big-web-simulation/` on port 3000, then:
```bash
python crawler.py seeds_test.txt --output crawled_faces --verbose
```

---

## Face Detection — OpenCV Haar Cascade

Both `FRWorker` (pre-crop before AdaFace) and `WebCrawler` (image filtering) use **OpenCV's built-in Haar Cascade classifier**:

- Classifier: `haarcascade_frontalface_default.xml` (ships with `opencv-python`)
- Algorithm: Viola-Jones boosted cascade of Haar-like features
- Runs entirely on **CPU**, no GPU or model download required
- Parameters used: `scaleFactor=1.1`, `minNeighbors=5`, `minSize=(30, 30)`
- Detects frontal faces only; profile or heavily angled faces may be missed
- Fast enough to process hundreds of images per minute on a standard CPU
