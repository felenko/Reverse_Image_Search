"""FRWorkerWithAlignment: index face images using RetinaFace + 3DDFA_V2 + ArcFace.

Scans a folder recursively, applies yaw-based alignment, and stores 512-d
ArcFace embeddings in LanceDB with alignment metadata.

Usage:
    python worker.py [input_folder] [--db face_db_aligned] [--device cpu|cuda]
                     [--tddfa-root PATH] [--max-faces N]

Examples:
    python worker.py templates/
    python worker.py templates/ --db face_db_aligned --device cuda
    python worker.py templates/ --tddfa-root ~/3DDFA_V2
"""
import argparse
import sys
from pathlib import Path

from database import open_table, insert_face
from pipeline import FacePipeline

DEFAULT_INPUT = Path(__file__).parent / "templates"
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".gif", ".tif", ".tiff",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index face images with alignment-aware ArcFace embeddings."
    )
    parser.add_argument(
        "input_folder", nargs="?", default=str(DEFAULT_INPUT),
        help=f"Folder to scan recursively (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--db", default="face_db_aligned",
        help="LanceDB directory (default: face_db_aligned)",
    )
    parser.add_argument(
        "--device", default=None, choices=["cpu", "cuda"],
        help="Inference device (default: auto-detect)",
    )
    parser.add_argument(
        "--tddfa-root", default=None, dest="tddfa_root",
        help="Path to 3DDFA_V2 repository root (enables 3D frontalization)",
    )
    parser.add_argument(
        "--max-faces", type=int, default=1, dest="max_faces",
        help="Max faces per image to index (default: 1 = largest detected face)",
    )
    parser.add_argument(
        "--det-thresh", type=float, default=0.5, dest="det_thresh",
        help="RetinaFace detection confidence threshold (default: 0.5)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_folder).resolve()
    if not input_path.is_dir():
        print(f"Error: folder not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    files = sorted(
        f for f in input_path.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        print(f"No image files found under {input_path}")
        return

    print(f"Found {len(files)} images under {input_path}")

    # Determine device
    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    print(f"Device: {device}")

    # Initialise components
    table    = open_table(args.db)
    pipeline = FacePipeline(
        device=device,
        det_thresh=args.det_thresh,
        tddfa_root=args.tddfa_root,
    )

    # Force model loading before the loop so the first image isn't slow
    print("Loading models…")
    _ = pipeline.detector
    _ = pipeline.aligner
    _ = pipeline.embedder
    print(f"Models ready. Frontalizer backend: {pipeline.aligner._frontalizer.backend}")

    # --- Indexing loop ---
    counts = {"ok": 0, "skip": 0, "error": 0}
    method_counts: dict[str, int] = {}

    print(f"\nIndexing -> {args.db}\n{'─'*60}")
    for i, fp in enumerate(files):
        rel = fp.relative_to(input_path)
        try:
            results = pipeline.process_bgr(
                __import__("cv2").imread(str(fp)),
                filename=str(rel).replace("\\", "/"),
                max_faces=args.max_faces,
            )
            if not results:
                print(f"  [skip]  no face detected  {rel}")
                counts["skip"] += 1
                continue

            for res in results:
                if res.embedding is None:
                    print(f"  [skip]  embedding failed  {rel}")
                    counts["skip"] += 1
                    continue

                row_id = insert_face(
                    table,
                    filename         = res.filename,
                    embedding        = res.embedding,
                    yaw              = res.yaw,
                    alignment_method = res.alignment_method,
                    confidence       = res.confidence,
                )
                method_counts[res.alignment_method] = (
                    method_counts.get(res.alignment_method, 0) + 1
                )
                counts["ok"] += 1
                print(
                    f"  [{i+1:>4}/{len(files)}]  "
                    f"id={row_id}  yaw={res.yaw:+6.1f}°  "
                    f"method={res.alignment_method:<14}  "
                    f"conf={res.confidence:.3f}  {rel}"
                )

        except Exception as exc:
            print(f"  [error] {rel}: {exc}", file=sys.stderr)
            counts["error"] += 1

    print(f"\n{'─'*60}")
    print(f"Done.  indexed={counts['ok']}  skipped={counts['skip']}  errors={counts['error']}")
    if method_counts:
        print("Alignment method breakdown:")
        for m, n in sorted(method_counts.items()):
            print(f"  {m:<18} {n:>5}")


if __name__ == "__main__":
    main()
