"""
FR worker: read face images from templates/ (and subfolders), compute AdaFace
embeddings, and store them in LanceDB (supports concurrent writers).
"""
import argparse
import sys
from pathlib import Path

from database import open_table, insert_face
from adaface_model import AdaFaceExtractor

DEFAULT_INPUT = Path(__file__).parent / "templates"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff", ".heic"}


def main():
    parser = argparse.ArgumentParser(description="FR worker: index face images into LanceDB with AdaFace embeddings.")
    parser.add_argument("input_folder", type=str, nargs="?", default=str(DEFAULT_INPUT),
                        help=f"Folder to scan recursively for face images (default: {DEFAULT_INPUT})")
    parser.add_argument("--db", type=str, default="face_db",
                        help="LanceDB directory path (default: face_db)")
    parser.add_argument("--cache-dir", type=str, default=None, help="Directory to cache AdaFace model")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Device for inference")
    args = parser.parse_args()

    input_path = Path(args.input_folder).resolve()
    if not input_path.is_dir():
        print(f"Error: input folder does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    table = open_table(args.db)

    extractor = AdaFaceExtractor(cache_dir=args.cache_dir, device=args.device)
    files = sorted(
        f for f in input_path.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        print(f"No image files found under {input_path}")
        return

    print(f"Processing {len(files)} images from {input_path} -> {args.db}")
    for i, fp in enumerate(files):
        try:
            emb = extractor.get_embedding_from_path(str(fp))
            if emb is None:
                print(f"  [skip] No face detected: {fp.relative_to(input_path)}")
                continue
            rel = fp.relative_to(input_path)
            image_file_name = str(rel).replace("\\", "/")
            row_id = insert_face(table, image_file_name, emb)
            print(f"  [{i+1}/{len(files)}] Indexed id={row_id} -> {image_file_name}")
        except Exception as e:
            print(f"  [error] {fp.relative_to(input_path)}: {e}", file=sys.stderr)
    print("Done.")


if __name__ == "__main__":
    main()
