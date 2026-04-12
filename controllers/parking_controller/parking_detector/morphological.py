from __future__ import annotations

import cv2
import numpy as np

from .base import DetectionBackend, SpotConfig, SpotResult

_KERNEL_3x3 = np.ones((3, 3), np.uint8)


class MorphologicalBackend(DetectionBackend):
    """
    Backend de detección basado en procesamiento morfológico.

    Pipeline idéntico al original de main.py:
      BGR → Grayscale → GaussianBlur(3,3,σ=1)
          → adaptiveThreshold(GAUSSIAN_C, INV, blockSize=25, C=16)
          → medianBlur(5) → dilate(3x3, iter=1)

    Clasificación: countNonZero(ROI) < free_threshold → libre.
    """

    def __init__(self, free_threshold: int = 1800) -> None:
        self.free_threshold = free_threshold

    def classify_spots(
        self,
        frame: np.ndarray,
        spots: list[SpotConfig],
    ) -> list[SpotResult]:
        processed = self._preprocess(frame)
        results: list[SpotResult] = []
        for spot in spots:
            roi = processed[spot.y : spot.y + spot.height, spot.x : spot.x + spot.width]
            count = cv2.countNonZero(roi)
            status: str = "free" if count < self.free_threshold else "occupied"
            results.append(SpotResult(global_id=spot.global_id, status=status))
        return results

    @staticmethod
    def _preprocess(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 1)
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            25, 16,
        )
        median = cv2.medianBlur(thresh, 5)
        dilated = cv2.dilate(median, _KERNEL_3x3, iterations=1)
        return dilated