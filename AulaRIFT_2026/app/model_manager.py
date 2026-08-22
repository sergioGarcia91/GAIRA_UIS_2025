# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 19:37:09 2026

@author: thesa
"""

from __future__ import annotations

import gc

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ultralytics import YOLO


@dataclass
class ModelInfo:
    path: str
    relative_name: str
    valid: bool

    task: str = "-"
    class_count: int = 0
    classes_preview: str = "-"

    inference_ok: bool = False

    error: str = ""


def find_pt_models(folder):
    """
    Busca todos los archivos .pt dentro de una carpeta
    y sus subcarpetas.
    """

    root = Path(folder)

    if not root.exists():
        return []

    models = list(root.rglob("*.pt"))

    models = [
        model
        for model in models
        if model.is_file()
    ]

    models.sort(key=lambda x: str(x).lower())

    return models


def extract_names(model):
    """
    Extrae los nombres de las clases de un modelo YOLO.
    """

    names = getattr(model, "names", None)

    if names is None:

        if getattr(model, "model", None) is not None:
            names = getattr(model.model, "names", None)

    if isinstance(names, dict):

        try:

            return [
                str(names[key])
                for key in sorted(names)
            ]

        except Exception:

            return list(
                map(str, names.values())
            )

    if isinstance(names, (list, tuple)):

        return list(map(str, names))

    return []


def validate_model(
    model_path,
    root_folder,
    run_test_inference=True,
):
    """
    Valida un modelo YOLO.

    Etapa 1:
    intenta cargar el archivo.

    Etapa 2:
    ejecuta una predicción sobre una imagen sintética.
    """

    model_path = Path(model_path)

    root_folder = Path(root_folder)

    try:

        relative_name = str(
            model_path.relative_to(root_folder)
        )

    except ValueError:

        relative_name = model_path.name

    model = None

    try:

        # -----------------------------
        # Cargar modelo
        # -----------------------------

        model = YOLO(
            str(model_path)
        )

        # -----------------------------
        # Tipo de modelo
        # -----------------------------

        task = str(
            getattr(
                model,
                "task",
                "unknown",
            )
        )

        # -----------------------------
        # Clases
        # -----------------------------

        class_names = extract_names(model)

        class_count = len(class_names)

        if class_names:

            preview = class_names[:5]

            classes_preview = ", ".join(preview)

            if len(class_names) > 5:

                classes_preview += ", ..."

        else:

            classes_preview = "-"

        # -----------------------------
        # Inferencia de prueba
        # -----------------------------

        inference_ok = False

        if run_test_inference:

            dummy_image = np.zeros(
                (640, 640, 3),
                dtype=np.uint8,
            )

            model.predict(
                source=dummy_image,
                device="cpu",
                verbose=False,
            )

            inference_ok = True

        else:

            inference_ok = True

        # -----------------------------
        # Resultado válido
        # -----------------------------

        return ModelInfo(
            path=str(model_path),
            relative_name=relative_name,
            valid=True,
            task=task,
            class_count=class_count,
            classes_preview=classes_preview,
            inference_ok=inference_ok,
            error="",
        )

    except Exception as error:

        # -----------------------------
        # Modelo con error
        # -----------------------------

        return ModelInfo(
            path=str(model_path),
            relative_name=relative_name,
            valid=False,
            error=f"{type(error).__name__}: {error}",
        )

    finally:

        if model is not None:

            try:
                del model

            except Exception:
                pass

        gc.collect()