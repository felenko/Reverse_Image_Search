"""
WebCrawler: JS-rendered page crawling via Playwright, OpenCV face detection,
saves face images to an output folder.

Usage:
    python crawler.py seeds.txt
    python crawler.py seeds.txt --output faces_out --max-pages 50 --verbose
"""
import argparse
import hashlib
import logging
import sys
import time
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import requests


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("crawler")


# ── Face detection ────────────────────────────────────────────────────────────

_CASCADE = None

def _get_cascade() -> cv2.CascadeClassifier:
    global _CASCADE
    if _CASCADE is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _CASCADE = cv2.CascadeClassifier(path)
    return _CASCADE


def has_face(img_bytes: bytes) -> bool:
    try:
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = _get_cascade().detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return len(faces) > 0
    except Exception:
        return False


# ── Page scraping via Playwright ──────────────────────────────────────────────

def scrape_page(page, url: str, wait_ms: int, log: logging.Logger):
    """
    Navigate to URL, wait for network idle + optional extra wait for JS,
    then extract all links and image src URLs from the fully-rendered DOM.
    Returns (links: list[str], images: list[str]).
    """
    page.goto(url, wait_until="networkidle", timeout=30_000)
    if wait_ms > 0:
        page.wait_for_timeout(wait_ms)

    links = page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
    images = page.eval_on_selector_all("img[src]", "els => els.map(e => e.src)")

    log.debug("    DOM: %d <a> tags, %d <img> tags", len(links), len(images))
    return links, images


# ── Main crawl loop ───────────────────────────────────────────────────────────

def crawl(seed_urls: list, output_dir: Path, max_pages: int,
          delay: float, wait_ms: int, log: logging.Logger):

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log.error("playwright not installed — run: pip install playwright && playwright install chromium")
        sys.exit(1)

    http = requests.Session()
    http.headers["User-Agent"] = "FaceCrawler/1.0"

    queue = deque(seed_urls)
    visited_pages: set = set()
    visited_images: set = set()

    stats = {"pages": 0, "images_checked": 0, "faces_saved": 0, "errors": 0}
    t_start = time.time()

    log.info("═" * 64)
    log.info("Crawler starting")
    log.info("  Seeds      : %d URL(s)", len(seed_urls))
    log.info("  Output     : %s", output_dir.resolve())
    log.info("  Max pages  : %s", max_pages or "unlimited")
    log.info("  Page delay : %.1fs  |  JS wait: %dms", delay, wait_ms)
    log.info("═" * 64)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) FaceCrawler/1.0"
        )
        page = context.new_page()

        while queue:
            if max_pages and stats["pages"] >= max_pages:
                log.info("Max pages limit (%d) reached — stopping.", max_pages)
                break

            url = queue.popleft()
            if url in visited_pages:
                continue
            visited_pages.add(url)

            log.info("── Page %d %s", stats["pages"] + 1, "─" * 40)
            log.info("  URL: %s", url)

            t_page = time.time()
            try:
                links, images = scrape_page(page, url, wait_ms, log)
            except Exception as e:
                log.warning("  SKIPPED — %s", e)
                stats["errors"] += 1
                continue

            stats["pages"] += 1

            # Queue new pages
            new_links = 0
            for href in links:
                p = urlparse(href)
                clean = p._replace(fragment="").geturl()
                if clean not in visited_pages and p.scheme in ("http", "https"):
                    queue.append(clean)
                    new_links += 1

            log.info("  Links  : %d on page / %d new queued / %d total in queue",
                     len(links), new_links, len(queue))
            log.info("  Images : %d img tags on page", len(images))

            # Process images
            page_new = 0
            page_faces = 0

            for img_url in images:
                if img_url in visited_images:
                    continue
                visited_images.add(img_url)
                page_new += 1

                if not img_url.startswith("http"):
                    log.debug("    SKIP non-http image: %s", img_url)
                    continue

                log.debug("    Fetching: %s", img_url)
                try:
                    r = http.get(img_url, timeout=15)
                    r.raise_for_status()
                    img_bytes = r.content
                except Exception as e:
                    log.warning("    FETCH ERROR %s — %s", img_url, e)
                    stats["errors"] += 1
                    continue

                size_kb = len(img_bytes) / 1024
                stats["images_checked"] += 1

                if not has_face(img_bytes):
                    log.debug("    No face (%.1f KB): %s", size_kb, img_url)
                    continue

                img_path = urlparse(img_url).path
                ext = (Path(img_path).suffix.split("?")[0] or ".jpg")
                stem = (Path(img_path).stem or "face")[:40]
                name_hash = hashlib.md5(img_url.encode()).hexdigest()[:8]
                filename = f"{stem}_{name_hash}{ext}"
                (output_dir / filename).write_bytes(img_bytes)

                stats["faces_saved"] += 1
                page_faces += 1
                log.info("    ✓ FACE saved: %-45s  %.1f KB", filename, size_kb)

            log.info("  Summary: %d new images / %d faces saved  (%.1fs)",
                     page_new, page_faces, time.time() - t_page)

            if delay:
                time.sleep(delay)

        browser.close()

    elapsed = time.time() - t_start
    log.info("═" * 64)
    log.info("CRAWL COMPLETE  (%.1fs)", elapsed)
    log.info("  Pages crawled   : %d", stats["pages"])
    log.info("  Images checked  : %d", stats["images_checked"])
    log.info("  Faces saved     : %d  →  %s", stats["faces_saved"], output_dir)
    log.info("  Errors          : %d", stats["errors"])
    log.info("═" * 64)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Web crawler: JS-rendered pages, face detection, saves face images."
    )
    parser.add_argument("seed_file",
                        help="Text file with one seed URL per line (# lines are comments)")
    parser.add_argument("--output", "-o", default="crawled_faces",
                        help="Output folder for face images (default: crawled_faces)")
    parser.add_argument("--max-pages", type=int, default=0,
                        help="Max pages to crawl, 0 = unlimited (default: 0)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds to wait between page loads (default: 0.5)")
    parser.add_argument("--wait", type=int, default=2000,
                        help="Extra ms to wait after page load for JS to settle (default: 2000)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show debug output (every image URL, face detection result)")
    args = parser.parse_args()

    log = setup_logging(args.verbose)

    seed_file = Path(args.seed_file)
    if not seed_file.is_file():
        log.error("Seed file not found: %s", seed_file)
        sys.exit(1)

    seeds = [
        line.strip()
        for line in seed_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not seeds:
        log.error("No URLs found in seed file (empty or all comments)")
        sys.exit(1)

    log.info("Seed file: %s  (%d URL(s) loaded)", seed_file, len(seeds))

    crawl(
        seed_urls=seeds,
        output_dir=Path(args.output),
        max_pages=args.max_pages,
        delay=args.delay,
        wait_ms=args.wait,
        log=log,
    )


if __name__ == "__main__":
    main()
