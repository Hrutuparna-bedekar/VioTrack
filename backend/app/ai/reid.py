"""
OSNet-based Re-Identification service.

Extracts compact 512-dim appearance embeddings from person crops using a
lightweight OSNet model (osnet_x0_25 — ~1.6MB, CPU-friendly).

Usage pattern:
  reid = ReIDService()
  emb = reid.extract(frame, bbox)          # numpy float32 [512]
  sim = reid.cosine_similarity(emb1, emb2) # float in [-1, 1]

The embeddings are session-scoped: they live in memory during video
processing and are serialised to BLOB in the DB per TrackedIndividual row
for persistent image-to-person search.
"""

from __future__ import annotations

import logging
import numpy as np
import cv2
from typing import Optional
import io

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OSNet input spec
# ---------------------------------------------------------------------------
_INPUT_SIZE = (128, 256)   # (width, height) — standard for person ReID


class ReIDService:
    """
    Wraps torchreid OSNet for person crop embedding extraction.

    The model is loaded lazily on first use so that the rest of the system
    starts even if torchreid is not installed (degrades gracefully).
    """

    def __init__(self):
        self._extractor = None
        self._available = False
        self._try_load()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _try_load(self):
        """Try to build the torchreid feature extractor. Fail silently."""
        try:
            import torchreid  # noqa: F401 — just check importability
            from torchreid.utils import FeatureExtractor
            self._extractor = FeatureExtractor(
                model_name="osnet_x0_25",   # lightest OSNet variant (~1.6 MB)
                model_path=None,             # downloads pretrained weights automatically
                device="cpu",               # use GPU string "cuda:0" if available
            )
            self._available = True
            logger.info("OSNet ReID model loaded successfully (osnet_x0_25)")
        except ImportError:
            logger.warning(
                "torchreid not installed — ReID disabled. "
                "Install with: pip install torchreid"
            )
        except Exception as exc:
            logger.warning(f"ReID model failed to load: {exc}")

    @property
    def available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def extract(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> Optional[np.ndarray]:
        """
        Extract a 512-dim L2-normalised embedding from a person crop.

        Args:
            frame: Full BGR frame (H×W×3 uint8).
            bbox:  (x1, y1, x2, y2) in pixel coordinates.

        Returns:
            float32 numpy array of shape (512,), or None on failure.
        """
        if not self._available:
            return None

        crop = self._safe_crop(frame, bbox)
        if crop is None:
            return None

        try:
            import torch
            # torchreid expects RGB PIL or BGR numpy; FeatureExtractor handles both
            features = self._extractor([crop])   # returns tensor (1, 512)
            emb = features[0].cpu().numpy().astype(np.float32)
            # L2-normalise so cosine similarity == dot product
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb
        except Exception as exc:
            logger.debug(f"ReID extraction failed: {exc}")
            return None

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Cosine similarity between two L2-normalised embeddings.
        Returns a float in [0, 1] (higher = more similar).
        Both inputs must already be L2-normalised (output of extract()).
        """
        if emb1 is None or emb2 is None:
            return 0.0
        sim = float(np.dot(emb1, emb2))
        # Clamp to [0,1] — negative values mean very different appearances
        return max(0.0, min(1.0, (sim + 1.0) / 2.0))

    # ------------------------------------------------------------------
    # Serialisation helpers  (for storing in SQLite BLOB)
    # ------------------------------------------------------------------

    @staticmethod
    def embedding_to_bytes(emb: Optional[np.ndarray]) -> Optional[bytes]:
        """Serialise embedding to bytes for DB storage."""
        if emb is None:
            return None
        buf = io.BytesIO()
        np.save(buf, emb)
        return buf.getvalue()

    @staticmethod
    def bytes_to_embedding(data: Optional[bytes]) -> Optional[np.ndarray]:
        """Deserialise embedding from DB bytes."""
        if not data:
            return None
        try:
            buf = io.BytesIO(data)
            return np.load(buf)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_crop(
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> Optional[np.ndarray]:
        """Crop and resize to OSNet input size with bounds checking."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(c) for c in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Resize to (W=128, H=256)
        crop = cv2.resize(crop, _INPUT_SIZE)
        return crop


# ---------------------------------------------------------------------------
# Module-level singleton — shared across the whole backend process
# ---------------------------------------------------------------------------
_reid_service: Optional[ReIDService] = None


def get_reid_service() -> ReIDService:
    """Return the global ReIDService singleton, creating it if needed."""
    global _reid_service
    if _reid_service is None:
        _reid_service = ReIDService()
    return _reid_service
