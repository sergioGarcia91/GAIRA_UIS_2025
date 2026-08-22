# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 06:49:35 2026

@author: thesa
"""

from __future__ import annotations

import csv
import gc

from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO


# =========================================================
# NOMBRES SEGUROS
# =========================================================

def safe_name(text):

    invalid_chars = '<>:"/\\|?*'

    for char in invalid_chars:

        text = text.replace(
            char,
            "_",
        )

    return text.strip()


# =========================================================
# CLASES
# =========================================================

def get_class_name(
    names,
    class_id,
):

    if isinstance(
        names,
        dict,
    ):

        return str(
            names.get(
                class_id,
                class_id,
            )
        )

    if isinstance(
        names,
        (list, tuple),
    ):

        if (
            0 <= class_id < len(names)
        ):

            return str(
                names[class_id]
            )

    return str(
        class_id
    )


# =========================================================
# EXTRAER RESULTADOS
# =========================================================

def cache_result(
    result,
    frame_shape,
):

    height, width = (
        frame_shape[:2]
    )

    cached = {

        "boxes": [],

        "masks": [],

        "classification": None,
    }

    # -----------------------------------------------------
    # Boxes
    # -----------------------------------------------------

    if result.boxes is not None:

        boxes = (
            result.boxes.xyxy
            .detach()
            .cpu()
            .numpy()
        )

        classes = (
            result.boxes.cls
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        )

        confidences = (
            result.boxes.conf
            .detach()
            .cpu()
            .numpy()
        )

        for i in range(
            len(boxes)
        ):

            cached["boxes"].append(
                {
                    "xyxy": boxes[i],
                    "class_id": int(
                        classes[i]
                    ),
                    "confidence": float(
                        confidences[i]
                    ),
                }
            )

    # -----------------------------------------------------
    # Máscaras
    # -----------------------------------------------------

    if result.masks is not None:

        masks = (
            result.masks.data
            .detach()
            .cpu()
            .numpy()
        )

        for mask in masks:

            mask_resized = cv2.resize(
                mask,
                (
                    width,
                    height,
                ),
                interpolation=cv2.INTER_NEAREST,
            )

            binary = (
                mask_resized > 0.5
            )

            cached["masks"].append(
                binary
            )

    # -----------------------------------------------------
    # Clasificación
    # -----------------------------------------------------

    if result.probs is not None:

        try:

            cached[
                "classification"
            ] = {

                "class_id": int(
                    result.probs.top1
                ),

                "confidence": float(
                    result.probs.top1conf
                    .detach()
                    .cpu()
                    .item()
                ),
            }

        except Exception:

            pass

    return cached


# =========================================================
# DIBUJAR RESULTADO
# =========================================================

def draw_cached_result(
    frame,
    cached,
    class_names,
):

    output = frame.copy()

    # -----------------------------------------------------
    # Máscaras
    # -----------------------------------------------------

    if cached["masks"]:

        overlay = output.copy()

        for mask in cached["masks"]:

            overlay[
                mask
            ] = (
                0,
                200,
                0,
            )

        output = cv2.addWeighted(
            output,
            0.70,
            overlay,
            0.30,
            0,
        )

    # -----------------------------------------------------
    # Boxes
    # -----------------------------------------------------

    for detection in cached[
        "boxes"
    ]:

        x1, y1, x2, y2 = (
            detection["xyxy"]
        )

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        class_id = (
            detection[
                "class_id"
            ]
        )

        confidence = (
            detection[
                "confidence"
            ]
        )

        class_name = (
            get_class_name(
                class_names,
                class_id,
            )
        )

        cv2.rectangle(
            output,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            (
                0,
                255,
                0,
            ),
            2,
        )

        label = (
            f"{class_name} "
            f"{confidence:.2f}"
        )

        cv2.putText(
            output,
            label,
            (
                x1,
                max(
                    20,
                    y1 - 10,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (
                0,
                255,
                0,
            ),
            2,
            cv2.LINE_AA,
        )

    # -----------------------------------------------------
    # Clasificación
    # -----------------------------------------------------

    classification = (
        cached["classification"]
    )

    if classification is not None:

        class_id = (
            classification[
                "class_id"
            ]
        )

        confidence = (
            classification[
                "confidence"
            ]
        )

        class_name = (
            get_class_name(
                class_names,
                class_id,
            )
        )

        cv2.putText(
            output,
            (
                f"{class_name}: "
                f"{confidence:.2f}"
            ),
            (
                20,
                40,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (
                0,
                255,
                0,
            ),
            2,
            cv2.LINE_AA,
        )

    return output


# =========================================================
# PROCESAR VIDEO CON UN MODELO
# =========================================================

def process_video_with_model(
    model_path,
    model_label,
    video_path,
    output_root,
    settings,
    inference_fps=1,
    progress_callback=None,
    interruption_callback=None,
):

    model_path = Path(
        model_path
    )

    video_path = Path(
        video_path
    )

    output_root = Path(
        output_root
    )

    # -----------------------------------------------------
    # Nombre del modelo
    # -----------------------------------------------------

    model_name = safe_name(
        Path(model_label).with_suffix("").as_posix()
    )

    model_folder = (
        output_root
        / model_name
    )

    model_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_video = (
        model_folder
        / "prediccion.mp4"
    )

    output_csv = (
        model_folder
        / "predicciones.csv"
    )

    cap = None
    writer = None
    model = None

    rows = []

    try:

        # =================================================
        # ABRIR VIDEO
        # =================================================

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():

            raise RuntimeError(
                "No fue posible abrir "
                "el video original."
            )

        source_fps = float(
            cap.get(
                cv2.CAP_PROP_FPS
            )
            or 0
        )

        if (
            source_fps <= 1
            or source_fps > 120
        ):

            source_fps = 30.0

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
            or 0
        )

        # =================================================
        # VIDEO DE SALIDA
        # =================================================

        fourcc = (
            cv2.VideoWriter_fourcc(
                *"mp4v"
            )
        )

        writer = cv2.VideoWriter(
            str(output_video),
            fourcc,
            source_fps,
            (
                width,
                height,
            ),
        )

        if not writer.isOpened():

            raise RuntimeError(
                "No fue posible crear "
                "el video de predicción."
            )

        # =================================================
        # MODELO
        # =================================================

        model = YOLO(
            str(model_path)
        )

        class_names = getattr(
            model,
            "names",
            {},
        )

        # =================================================
        # PARÁMETROS
        # =================================================

        device = settings.get(
            "device",
            "auto",
        )

        kwargs = {

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

        if device != "auto":

            kwargs[
                "device"
            ] = device

        use_half = (
            settings.get(
                "half",
                False,
            )
        )

        if use_half:

            gpu_possible = (
                torch.cuda.is_available()
                and device != "cpu"
            )

            if gpu_possible:

                kwargs[
                    "half"
                ] = True

        # =================================================
        # FRECUENCIA YOLO
        # =================================================

        if inference_fps > 0:

            inference_interval = (
                1.0
                / float(
                    inference_fps
                )
            )

        else:

            inference_interval = 0

        next_inference_time = 0.0

        # Sin detecciones al inicio
        cached = {

            "boxes": [],

            "masks": [],

            "classification": None,
        }

        frame_index = 0

        # =================================================
        # LOOP
        # =================================================

        while True:

            if (
                interruption_callback
                is not None
            ):

                if interruption_callback():

                    break

            success, frame = (
                cap.read()
            )

            if not success:

                break

            video_time = (
                frame_index
                / source_fps
            )

            # ---------------------------------------------
            # ¿Inferir este frame?
            # ---------------------------------------------

            if inference_interval == 0:

                should_infer = True

            else:

                should_infer = (
                    video_time
                    + 1e-6
                    >= next_inference_time
                )

            # ---------------------------------------------
            # YOLO
            # ---------------------------------------------

            if should_infer:

                with torch.inference_mode():

                    results = model.predict(
                        source=frame,
                        **kwargs,
                    )

                if results:

                    result = results[0]

                    cached = cache_result(
                        result,
                        frame.shape,
                    )

                    # =====================================
                    # CSV
                    # =====================================

                    boxes = (
                        cached["boxes"]
                    )

                    masks = (
                        cached["masks"]
                    )

                    for i, detection in enumerate(
                        boxes
                    ):

                        class_id = (
                            detection[
                                "class_id"
                            ]
                        )

                        confidence = (
                            detection[
                                "confidence"
                            ]
                        )

                        x1, y1, x2, y2 = (
                            detection[
                                "xyxy"
                            ]
                        )

                        mask_area = 0

                        if i < len(
                            masks
                        ):

                            mask_area = int(
                                masks[i].sum()
                            )

                        rows.append(
                            {
                                "tiempo_s": round(
                                    video_time,
                                    3,
                                ),
                                "frame": (
                                    frame_index
                                ),
                                "clase_id": (
                                    class_id
                                ),
                                "clase": (
                                    get_class_name(
                                        class_names,
                                        class_id,
                                    )
                                ),
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
                                "area_mascara_px": (
                                    mask_area
                                ),
                            }
                        )

                if inference_interval > 0:

                    while (
                        next_inference_time
                        <= video_time
                    ):

                        next_inference_time += (
                            inference_interval
                        )

            # ---------------------------------------------
            # Dibujar última predicción
            # ---------------------------------------------

            annotated = (
                draw_cached_result(
                    frame,
                    cached,
                    class_names,
                )
            )

            writer.write(
                annotated
            )

            frame_index += 1

            # ---------------------------------------------
            # Progreso
            # ---------------------------------------------

            if (
                progress_callback
                is not None
                and total_frames > 0
            ):

                progress_callback(
                    frame_index,
                    total_frames,
                )

        # =================================================
        # CSV
        # =================================================

        fieldnames = [
            "tiempo_s",
            "frame",
            "clase_id",
            "clase",
            "confianza",
            "x1",
            "y1",
            "x2",
            "y2",
            "area_mascara_px",
        ]

        with open(
            output_csv,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:

            writer_csv = (
                csv.DictWriter(
                    file,
                    fieldnames=fieldnames,
                )
            )

            writer_csv.writeheader()

            writer_csv.writerows(
                rows
            )

        return {

            "model": model_label,

            "video": str(
                output_video
            ),

            "csv": str(
                output_csv
            ),

            "detections": len(
                rows
            ),
        }

    finally:

        if cap is not None:

            cap.release()

        if writer is not None:

            writer.release()

        if model is not None:

            del model

        gc.collect()

        # ---------------------------------------------
        # Liberar GPU entre modelos
        # ---------------------------------------------

        if torch.cuda.is_available():

            try:

                torch.cuda.empty_cache()

            except Exception:

                pass