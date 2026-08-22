# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 19:47:13 2026

@author: thesa
"""

from __future__ import annotations

import csv
import gc
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def safe_name(text):
    """
    Limpia un nombre para poder utilizarlo como carpeta.
    """

    invalid_chars = '<>:"/\\|?*'

    for char in invalid_chars:
        text = text.replace(char, "_")

    return text.strip()


def process_image_with_model(
    model_path,
    image_path,
    output_folder,
    settings,
):
    """
    Ejecuta un modelo YOLO sobre una imagen.

    Guarda:
    - imagen anotada
    - CSV de predicciones
    - máscaras individuales si el modelo es segment
    """

    model_path = Path(model_path)
    image_path = Path(image_path)
    output_folder = Path(output_folder)

    # ---------------------------------------------------------
    # Crear carpeta correspondiente al modelo
    # ---------------------------------------------------------

    model_name = safe_name(
        model_path.stem
    )

    model_output = (
        output_folder
        / model_name
    )

    model_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = None

    try:

        # -----------------------------------------------------
        # Cargar modelo
        # -----------------------------------------------------

        model = YOLO(
            str(model_path)
        )

        # -----------------------------------------------------
        # Dispositivo
        # -----------------------------------------------------

        device = settings.get(
            "device",
            "auto",
        )

        if device == "auto":
            device = None

        # -----------------------------------------------------
        # Parámetros
        # -----------------------------------------------------

        predict_kwargs = {
            "source": str(image_path),
            "conf": settings.get(
                "conf",
                0.25,
            ),
            "iou": settings.get(
                "iou",
                0.70,
            ),
            "imgsz": settings.get(
                "imgsz",
                640,
            ),
            "max_det": settings.get(
                "max_det",
                300,
            ),
            "verbose": False,
        }

        if device is not None:
            predict_kwargs["device"] = device

        # FP16 solamente cuando se solicita
        if settings.get(
            "half",
            False,
        ):

            predict_kwargs["half"] = True

        # -----------------------------------------------------
        # Predicción
        # -----------------------------------------------------

        results = model.predict(
            **predict_kwargs
        )

        if not results:

            raise RuntimeError(
                "El modelo no devolvió resultados."
            )

        result = results[0]

        # -----------------------------------------------------
        # Guardar imagen anotada
        # -----------------------------------------------------

        annotated = result.plot()

        annotated_path = (
            model_output
            / "prediccion.jpg"
        )

        cv2.imwrite(
            str(annotated_path),
            annotated,
        )

        # -----------------------------------------------------
        # CSV
        # -----------------------------------------------------

        csv_path = (
            model_output
            / "predicciones.csv"
        )

        rows = []

        names = result.names

        boxes = result.boxes
        masks = result.masks

        # -----------------------------------------------------
        # Detección / segmentación
        # -----------------------------------------------------

        if boxes is not None:

            xyxy = (
                boxes.xyxy
                .cpu()
                .numpy()
            )

            classes = (
                boxes.cls
                .cpu()
                .numpy()
                .astype(int)
            )

            confidences = (
                boxes.conf
                .cpu()
                .numpy()
            )

            # -------------------------------------------------
            # Máscaras
            # -------------------------------------------------

            mask_data = None

            if masks is not None:

                mask_data = (
                    masks.data
                    .cpu()
                    .numpy()
                )

                masks_folder = (
                    model_output
                    / "mascaras"
                )

                masks_folder.mkdir(
                    exist_ok=True
                )

            # -------------------------------------------------
            # Cada objeto
            # -------------------------------------------------

            for i in range(
                len(xyxy)
            ):

                x1, y1, x2, y2 = (
                    xyxy[i]
                )

                class_id = int(
                    classes[i]
                )

                class_name = names.get(
                    class_id,
                    str(class_id),
                )

                confidence = float(
                    confidences[i]
                )

                mask_pixels = 0

                mask_file = ""

                # ---------------------------------------------
                # Guardar máscara
                # ---------------------------------------------

                if (
                    mask_data is not None
                    and i < len(mask_data)
                ):

                    mask = mask_data[i]

                    binary_mask = (
                        mask > 0.5
                    ).astype(
                        np.uint8
                    )

                    mask_pixels = int(
                        binary_mask.sum()
                    )

                    mask_file = (
                        f"mask_{i + 1:03d}_"
                        f"{safe_name(class_name)}.png"
                    )

                    mask_path = (
                        masks_folder
                        / mask_file
                    )

                    cv2.imwrite(
                        str(mask_path),
                        binary_mask * 255,
                    )

                rows.append(
                    {
                        "objeto": i + 1,
                        "clase_id": class_id,
                        "clase": class_name,
                        "confianza": round(
                            confidence,
                            6,
                        ),
                        "x1": round(
                            float(x1),
                            2,
                        ),
                        "y1": round(
                            float(y1),
                            2,
                        ),
                        "x2": round(
                            float(x2),
                            2,
                        ),
                        "y2": round(
                            float(y2),
                            2,
                        ),
                        "area_bbox_px": round(
                            float(
                                (x2 - x1)
                                * (y2 - y1)
                            ),
                            2,
                        ),
                        "area_mascara_px": (
                            mask_pixels
                        ),
                        "archivo_mascara": (
                            mask_file
                        ),
                    }
                )

        # -----------------------------------------------------
        # Guardar CSV
        # -----------------------------------------------------

        fieldnames = [
            "objeto",
            "clase_id",
            "clase",
            "confianza",
            "x1",
            "y1",
            "x2",
            "y2",
            "area_bbox_px",
            "area_mascara_px",
            "archivo_mascara",
        ]

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            writer.writerows(
                rows
            )

        return {
            "model": model_name,
            "detections": len(rows),
            "annotated_image": str(
                annotated_path
            ),
            "csv": str(
                csv_path
            ),
            "output_folder": str(
                model_output
            ),
        }

    finally:

        if model is not None:

            del model

        gc.collect()