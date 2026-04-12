from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class SpotConfig:
    global_id: int
    x: int
    y: int
    width: int
    height: int


@dataclass
class SpotResult:
    global_id: int
    status: Literal["free", "occupied"]


class DetectionBackend(ABC):
    @abstractmethod
    def classify_spots(
        self,
        frame: np.ndarray,
        spots: list[SpotConfig],
    ) -> list[SpotResult]:
        """
        Analiza un frame BGR y devuelve el estado de cada plaza.

        Parameters
        ----------
        frame : np.ndarray
            Frame BGR completo (H x W x 3, uint8).
        spots : list[SpotConfig]
            Plazas de la cámara que envió el frame.

        Returns
        -------
        list[SpotResult]
            Un resultado por plaza, en el mismo orden que `spots`.
        """
        ...