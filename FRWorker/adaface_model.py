"""Load AdaFace (CVLFace IR50) and extract face embeddings. Uses OpenCV for face detection."""
import os
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image


# AdaFace expects 112x112 RGB input, normalized (x/255 - 0.5)/0.5
INPUT_SIZE = 112


def _download_and_load_adaface(cache_dir: Path):
    """Download AdaFace from Hugging Face and load with trust_remote_code."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_id = "minchul/cvlface_adaface_ir50_ms1mv2"

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("Install: pip install huggingface_hub")

    files_txt = cache_dir / "files.txt"
    if not files_txt.exists():
        hf_hub_download(repo_id, "files.txt", local_dir=cache_dir, local_dir_use_symlinks=False)
    with open(files_txt) as f:
        files = [line.strip() for line in f if line.strip()]
    for name in files + ["config.json", "wrapper.py", "model.safetensors"]:
        path = cache_dir / name
        if not path.exists():
            hf_hub_download(repo_id, name, local_dir=cache_dir, local_dir_use_symlinks=False)

    sys.path.insert(0, str(cache_dir))
    cwd = os.getcwd()
    try:
        os.chdir(cache_dir)
        from transformers import AutoModel
        model = AutoModel.from_pretrained(str(cache_dir), trust_remote_code=True,
                                          low_cpu_mem_usage=False)
    finally:
        os.chdir(cwd)
        sys.path.pop(0)
    return model


def _detect_face_opencv(img: np.ndarray) -> Optional[tuple]:
    """Return (x, y, w, h) of largest face or None. img is BGR."""
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    # Pick largest face
    (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
    return (x, y, w, h)


class AdaFaceExtractor:
    def __init__(self, cache_dir: Optional[str] = None, device: Optional[str] = None):
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cvlface_cache/adaface_ir50_ms1mv2")
        self.cache_dir = Path(cache_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _download_and_load_adaface(self.cache_dir)
            self._model.to(self.device)
            self._model.eval()
        return self._model

    def _align_face(self, img: Image.Image) -> Optional[torch.Tensor]:
        """Detect face with OpenCV, crop and resize to 112x112. Returns tensor (1,3,112,112) or None."""
        arr = np.array(img)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        box = _detect_face_opencv(arr)
        if box is None:
            return None
        x, y, w, h = box
        margin = 0.2
        x1 = max(0, int(x - margin * w))
        y1 = max(0, int(y - margin * h))
        x2 = min(arr.shape[1], int(x + w + margin * w))
        y2 = min(arr.shape[0], int(y + h + margin * h))
        face = arr[y1:y2, x1:x2]
        face = cv2.resize(face, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        # (H,W,3) [0,255] -> (1,3,112,112) normalized (x/255 - 0.5)/0.5
        face = (face.astype(np.float32) / 255.0 - 0.5) / 0.5
        face = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        return face

    def get_embedding_from_image(self, img: Image.Image) -> Optional[List[float]]:
        """Extract single face embedding from PIL Image. Returns 512-d list or None if no face."""
        if img.mode != "RGB":
            img = img.convert("RGB")
        face_batch = self._align_face(img)
        if face_batch is None:
            return None
        with torch.no_grad():
            out = self.model(face_batch)
        # AdaFace often returns (feature, norm); we use normalized feature for similarity
        if isinstance(out, (tuple, list)):
            feat = out[0]
        else:
            feat = out
        if feat.dim() == 2:
            feat = feat[0]
        emb = feat.cpu().float().numpy()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.tolist()

    def get_embedding_from_path(self, path: str) -> Optional[List[float]]:
        img = Image.open(path).convert("RGB")
        return self.get_embedding_from_image(img)
