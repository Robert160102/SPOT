from __future__ import annotations

from typing import Literal

import numpy as np

from .base import DetectionBackend, SpotConfig, SpotResult

# ---------------------------------------------------------------------------
# Constantes de clases
# ---------------------------------------------------------------------------

# COCO: detección de vehículos (modo "vehicle")
_VEHICLE_CLASSES: dict[int, str] = {2: "car", 5: "bus", 7: "truck"}

# VisDrone: detección de vehículos desde vista aérea (modo "visdrone")
_VISDRONE_VEHICLE_CLASSES: dict[int, str] = {3: "car", 4: "van", 5: "truck", 8: "bus"}

# PKLot: detección directa de plazas (modo "spot")
_PKLOT_OCCUPIED = "space-occupied"
_PKLOT_EMPTY = "space-empty"


class YOLOBackend(DetectionBackend):
    """
    Backend de detección basado en YOLO (ultralytics).

    Soporta dos modos de operación:

    mode="vehicle"  (COCO, default)
        Detecta vehículos (car/bus/truck) en el frame completo y usa IoA
        (Intersection over Spot Area) para decidir si una plaza está ocupada.
        Funciona con yolov8n.pt u otro modelo COCO.
        Limitación: COCO no fue entrenado con vistas cenitales — fiabilidad
        baja en cámaras que miran verticalmente hacia abajo.

    mode="spot"  (PKLot)
        El modelo detecta directamente las plazas como 'space-empty' o
        'space-occupied'. Para cada plaza configurada, busca la detección
        con mayor IoU y toma su clase como resultado.
        Usar con un modelo entrenado en PKLot.

    Requisito: pip install ultralytics
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        mode: Literal["vehicle", "visdrone", "spot"] = "vehicle",
        ioa_threshold: float = 0.4,
        iou_threshold: float = 0.3,
        confidence_threshold: float = 0.4,
        device: str = "cpu",
    ) -> None:
        """
        Parameters
        ----------
        model_path : str
            Ruta al modelo YOLO. Se descarga automáticamente si no existe
            (solo para modelos oficiales ultralytics).
        mode : "vehicle" | "visdrone" | "spot"
            Modo de detección.
            "vehicle"  → COCO (car/bus/truck). Para cámaras oblicuas.
            "visdrone" → VisDrone (car/van/truck/bus). Para vistas aéreas.
            "spot"     → PKLot (space-empty/space-occupied). Detecta plazas directamente.
        ioa_threshold : float
            Solo en modo "vehicle". Fracción mínima de plaza cubierta
            por el vehículo para considerarla ocupada.
        iou_threshold : float
            Solo en modo "spot". IoU mínima entre una detección PKLot y
            una plaza configurada para aceptar el match.
        confidence_threshold : float
            Confianza mínima de detección YOLO.
        device : str
            Dispositivo de inferencia: "cpu", "cuda", "cuda:0", "mps", …
        """
        from ultralytics import YOLO  # import tardío: no rompe sin ultralytics

        self._model = YOLO(model_path)
        self._model.to(device)
        self.mode = mode
        self.ioa_threshold = ioa_threshold
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold

    # ------------------------------------------------------------------
    # API pública (también usada por main.py para visualización)
    # ------------------------------------------------------------------

    def detect_vehicles(self, frame: np.ndarray) -> list[dict]:
        """
        Modos "vehicle" y "visdrone". Devuelve los vehículos detectados.

        Returns
        -------
        list de dicts:
            "bbox"       : (x1, y1, x2, y2)
            "confidence" : float
            "class_id"   : int
            "class_name" : str
        """
        class_map = (
            _VISDRONE_VEHICLE_CLASSES if self.mode == "visdrone" else _VEHICLE_CLASSES
        )
        results = self._model(frame, verbose=False)[0]
        vehicles: list[dict] = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id not in class_map or conf < self.confidence_threshold:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            vehicles.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
                "class_id": cls_id,
                "class_name": class_map[cls_id],
            })
        return vehicles

    def diagnose_frame(self, frame: np.ndarray) -> dict:
        """
        Diagnóstico completo de un frame: muestra todo lo que el modelo
        detecta ANTES de cualquier filtro de clase o confianza.
        Útil en modo debug para entender por qué no se detecta nada.

        Returns
        -------
        dict con:
            "model_classes"      : dict {id: name} de todas las clases del modelo
            "frame_shape"        : (h, w, c)
            "confidence_threshold": umbral configurado
            "mode"               : "vehicle" | "spot"
            "all_raw"            : todas las detecciones sin filtrar
            "after_conf_filter"  : detecciones que pasan la confianza mínima
            "after_class_filter" : detecciones que pasan también el filtro de clase
        """
        results = self._model(frame, verbose=False)[0]
        model_classes = results.names  # {0: "class_a", 1: "class_b", ...}

        all_raw = []
        after_conf = []
        after_class = []

        if self.mode == "visdrone":
            valid_names = set(_VISDRONE_VEHICLE_CLASSES.values())
        elif self.mode == "vehicle":
            valid_names = set(_VEHICLE_CLASSES.values())
        else:
            valid_names = {_PKLOT_OCCUPIED, _PKLOT_EMPTY}

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model_classes.get(cls_id, f"cls_{cls_id}")
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            entry = {"class_id": cls_id, "class_name": cls_name,
                     "confidence": round(conf, 3), "bbox": [x1, y1, x2, y2]}
            all_raw.append(entry)
            if conf >= self.confidence_threshold:
                after_conf.append(entry)
                if cls_name in valid_names or cls_id in _VEHICLE_CLASSES or cls_id in _VISDRONE_VEHICLE_CLASSES:
                    after_class.append(entry)

        return {
            "model_classes": model_classes,
            "frame_shape": list(frame.shape),
            "confidence_threshold": self.confidence_threshold,
            "mode": self.mode,
            "total_raw_detections": len(all_raw),
            "after_conf_filter": len(after_conf),
            "after_class_filter": len(after_class),
            "all_raw": all_raw,
        }

    def detect_spots_raw(self, frame: np.ndarray) -> list[dict]:
        """
        Solo en modo "spot" (PKLot). Devuelve todas las detecciones crudas
        del modelo de plazas.

        Returns
        -------
        list de dicts:
            "bbox"       : (x1, y1, x2, y2)
            "confidence" : float
            "class_name" : str  ("space-empty" | "space-occupied")
        """
        results = self._model(frame, verbose=False)[0]
        detections: list[dict] = []
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.confidence_threshold:
                continue
            cls_name = results.names[int(box.cls[0])]
            if cls_name not in (_PKLOT_OCCUPIED, _PKLOT_EMPTY):
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
                "class_name": cls_name,
            })
        return detections

    # ------------------------------------------------------------------
    # DetectionBackend interface
    # ------------------------------------------------------------------

    def classify_spots(
        self,
        frame: np.ndarray,
        spots: list[SpotConfig],
    ) -> list[SpotResult]:
        if self.mode == "spot":
            return self._classify_spots_pklot(frame, spots)
        return self._classify_spots_vehicle(frame, spots)  # vehicle y visdrone

    # ------------------------------------------------------------------
    # Modo "vehicle": IoA entre vehículos COCO y plazas
    # ------------------------------------------------------------------

    def _classify_spots_vehicle(
        self,
        frame: np.ndarray,
        spots: list[SpotConfig],
    ) -> list[SpotResult]:
        vehicles = self.detect_vehicles(frame)
        results: list[SpotResult] = []
        for spot in spots:
            spot_box = (spot.x, spot.y, spot.x + spot.width, spot.y + spot.height)
            occupied = any(
                self._compute_ioa(spot_box, v["bbox"]) >= self.ioa_threshold
                for v in vehicles
            )
            results.append(SpotResult(
                global_id=spot.global_id,
                status="occupied" if occupied else "free",
            ))
        return results

    # ------------------------------------------------------------------
    # Modo "spot": matching PKLot detections → plazas configuradas
    # ------------------------------------------------------------------

    def _classify_spots_pklot(
        self,
        frame: np.ndarray,
        spots: list[SpotConfig],
    ) -> list[SpotResult]:
        """
        Para cada plaza configurada busca la detección PKLot con mayor IoU.
        Si el mejor match supera iou_threshold y es 'space-occupied' → ocupada.
        Si no hay match suficiente → libre (asumimos plaza vacía no detectada).
        """
        detections = self.detect_spots_raw(frame)
        results: list[SpotResult] = []

        for spot in spots:
            spot_box = (spot.x, spot.y, spot.x + spot.width, spot.y + spot.height)
            best_iou = 0.0
            best_class = _PKLOT_EMPTY

            for det in detections:
                iou = self._compute_iou(spot_box, det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_class = det["class_name"]

            if best_iou >= self.iou_threshold and best_class == _PKLOT_OCCUPIED:
                status = "occupied"
            else:
                status = "free"

            results.append(SpotResult(global_id=spot.global_id, status=status))

        return results

    # ------------------------------------------------------------------
    # Métricas de solapamiento
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_ioa(
        spot_box: tuple[int, int, int, int],
        vehicle_box: tuple[int, int, int, int],
    ) -> float:
        """
        Intersection over Spot Area — modo vehicle.
        Mide qué fracción del área de la plaza está cubierta por el vehículo.
        """
        ix1 = max(spot_box[0], vehicle_box[0])
        iy1 = max(spot_box[1], vehicle_box[1])
        ix2 = min(spot_box[2], vehicle_box[2])
        iy2 = min(spot_box[3], vehicle_box[3])

        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        spot_area = (spot_box[2] - spot_box[0]) * (spot_box[3] - spot_box[1])
        return inter / spot_area if spot_area > 0 else 0.0

    @staticmethod
    def _compute_iou(
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> float:
        """
        Intersection over Union — modo spot (PKLot matching).
        """
        ix1 = max(box_a[0], box_b[0])
        iy1 = max(box_a[1], box_b[1])
        ix2 = min(box_a[2], box_b[2])
        iy2 = min(box_a[3], box_b[3])

        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0
