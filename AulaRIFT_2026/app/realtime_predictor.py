# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 06:03:27 2026

@author: thesa
"""

from __future__ import annotations

import gc
import threading
import time

import cv2
import numpy as np
import torch

from PySide6.QtCore import QThread, Signal
from ultralytics import YOLO

from app.camera_manager import open_camera


class RealtimePredictionThread(QThread):

    status_message = Signal(str)

    error_message = Signal(str)

    realtime_started = Signal()

    realtime_stopped = Signal()

    metrics_updated = Signal(
        float,   # FPS cámara
        float,   # FPS inferencia
        float,   # tiempo inferencia ms
    )

    recording_started = Signal(str)

    recording_time_updated = Signal(float)

    recording_finished = Signal(
        str,     # ruta del video
        float,   # duración
    )

    recording_error = Signal(str)

    def __init__(
        self,
        camera_index,
        model_path,
        settings,
        inference_fps=1,
        parent=None,
    ):

        super().__init__(parent)

        self.camera_index = camera_index

        self.model_path = model_path

        self.settings = settings

        # 0 = inferir todos los frames
        self.inference_fps = inference_fps

        self.running = False

        # ---------------------------------------------
        # SOLO se mantiene un frame de visualización
        # ---------------------------------------------

        self._frame_lock = threading.Lock()

        self._latest_frame = None

        # ---------------------------------------------
        # Predicción anterior
        # ---------------------------------------------

        self.cached_boxes = []

        self.cached_masks = []

        self.cached_classification = None

        self.class_names = {}

        # =====================================================
        # GRABACIÓN
        # =====================================================

        self._record_lock = threading.Lock()

        # Solicitud pendiente de grabación
        self._record_request = None

        # Solicitud para detener grabación
        self._record_stop_requested = False

        # VideoWriter
        self._video_writer = None

        self._recording_path = None

        self._recording_start_time = None

        self._recording_max_seconds = 30.0

        self._last_recording_signal_time = 0


    # =========================================================
    # CONTROL
    # =========================================================

    def stop(self):

        self.running = False


    def get_latest_frame(self):

        """
        Devuelve únicamente el último frame disponible.

        No existe una cola de frames.
        """

        with self._frame_lock:

            if self._latest_frame is None:

                return None

            return self._latest_frame.copy()


    def set_latest_frame(
        self,
        frame,
    ):

        with self._frame_lock:

            self._latest_frame = frame

    # =========================================================
    # NOMBRES DE CLASE
    # =========================================================

    def get_class_name(
        self,
        class_id,
    ):

        if isinstance(
            self.class_names,
            dict,
        ):

            return str(
                self.class_names.get(
                    class_id,
                    class_id,
                )
            )

        if isinstance(
            self.class_names,
            (list, tuple),
        ):

            if 0 <= class_id < len(
                self.class_names
            ):

                return str(
                    self.class_names[
                        class_id
                    ]
                )

        return str(
            class_id
        )

    # =========================================================
    # GUARDAR RESULTADO YOLO
    # =========================================================

    def cache_result(
        self,
        result,
        frame_shape,
    ):

        """
        Guarda únicamente arrays pequeños necesarios
        para dibujar la última predicción.

        NO guarda el objeto Results completo.
        """

        self.cached_boxes = []

        self.cached_masks = []

        self.cached_classification = None

        height, width = frame_shape[:2]

        # -----------------------------------------------------
        # DETECCIÓN / SEGMENTACIÓN
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

            for index in range(
                len(boxes)
            ):

                self.cached_boxes.append(
                    {
                        "xyxy": boxes[index],
                        "class_id": int(
                            classes[index]
                        ),
                        "confidence": float(
                            confidences[index]
                        ),
                    }
                )

        # -----------------------------------------------------
        # MÁSCARAS
        # -----------------------------------------------------

        if result.masks is not None:

            masks = (
                result.masks.data
                .detach()
                .cpu()
                .numpy()
            )

            for mask in masks:

                # Aseguramos tamaño igual al video
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

                self.cached_masks.append(
                    binary
                )

        # -----------------------------------------------------
        # CLASIFICACIÓN
        # -----------------------------------------------------

        if result.probs is not None:

            try:

                class_id = int(
                    result.probs.top1
                )

                confidence = float(
                    result.probs.top1conf
                    .detach()
                    .cpu()
                    .item()
                )

                self.cached_classification = {
                    "class_id": class_id,
                    "confidence": confidence,
                }

            except Exception:

                self.cached_classification = None

    # =========================================================
    # DIBUJAR PREDICCIÓN ANTERIOR SOBRE FRAME ACTUAL
    # =========================================================

    def draw_cached_predictions(
        self,
        frame,
    ):

        output = frame.copy()

        # -----------------------------------------------------
        # Máscaras
        # -----------------------------------------------------

        if self.cached_masks:

            overlay = output.copy()

            for index, mask in enumerate(
                self.cached_masks
            ):

                if mask.shape[:2] != output.shape[:2]:

                    mask = cv2.resize(
                        mask.astype(
                            np.uint8
                        ),
                        (
                            output.shape[1],
                            output.shape[0],
                        ),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)

                # Color simple para visualización
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
        # Bounding boxes
        # -----------------------------------------------------

        for detection in self.cached_boxes:

            x1, y1, x2, y2 = (
                detection[
                    "xyxy"
                ]
            )

            class_id = detection[
                "class_id"
            ]

            confidence = detection[
                "confidence"
            ]

            class_name = (
                self.get_class_name(
                    class_id
                )
            )

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

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

        if (
            self.cached_classification
            is not None
        ):

            class_id = (
                self.cached_classification[
                    "class_id"
                ]
            )

            confidence = (
                self.cached_classification[
                    "confidence"
                ]
            )

            class_name = (
                self.get_class_name(
                    class_id
                )
            )

            label = (
                f"{class_name}: "
                f"{confidence:.2f}"
            )

            cv2.putText(
                output,
                label,
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
    # THREAD PRINCIPAL
    # =========================================================

    def run(self):

        cap = None

        model = None

        try:

            # -------------------------------------------------
            # Abrir cámara
            # -------------------------------------------------

            self.status_message.emit(
                "Abriendo cámara..."
            )

            cap, backend = open_camera(
                self.camera_index
            )

            if cap is None:

                raise RuntimeError(
                    f"No fue posible abrir "
                    f"la cámara {self.camera_index}."
                )

            self.status_message.emit(
                f"Cámara {self.camera_index} abierta "
                f"con {backend}"
            )

            # -------------------------------------------------
            # Cargar modelo SOLO una vez
            # -------------------------------------------------

            self.status_message.emit(
                "Cargando modelo YOLO..."
            )

            model = YOLO(
                self.model_path
            )

            self.class_names = (
                getattr(
                    model,
                    "names",
                    {},
                )
            )

            # -------------------------------------------------
            # Parámetros YOLO
            # -------------------------------------------------

            device = self.settings.get(
                "device",
                "auto",
            )

            if device == "auto":

                device = None

            prediction_kwargs = {

                "conf": self.settings.get(
                    "conf",
                    0.25,
                ),

                "iou": self.settings.get(
                    "iou",
                    0.70,
                ),

                "imgsz": self.settings.get(
                    "imgsz",
                    640,
                ),

                "max_det": self.settings.get(
                    "max_det",
                    300,
                ),

                "verbose": False,
            }

            if device is not None:

                prediction_kwargs[
                    "device"
                ] = device

            # FP16 no debe usarse en CPU
            if self.settings.get(
                "half",
                False,
            ):

                if device != "cpu":

                    prediction_kwargs[
                        "half"
                    ] = True

            # -------------------------------------------------
            # Frecuencia de inferencia
            # -------------------------------------------------

            if self.inference_fps > 0:

                inference_interval = (
                    1.0
                    / self.inference_fps
                )

            else:

                # Cada frame
                inference_interval = 0

            last_inference_time = 0

            previous_inference_time = None

            measured_inference_fps = 0

            inference_ms = 0

            # -------------------------------------------------
            # FPS cámara
            # -------------------------------------------------

            fps_counter = 0

            fps_start = (
                time.perf_counter()
            )

            camera_fps = 0

            self.running = True

            self.realtime_started.emit()

            self.status_message.emit(
                "Predicción en tiempo real iniciada"
            )

            # =================================================
            # LOOP
            # =================================================

            while self.running:

                success, frame = (
                    cap.read()
                )

                if not success:

                    time.sleep(
                        0.01
                    )

                    continue

                fps_counter += 1

                current_time = (
                    time.perf_counter()
                )

                # =================================================
                # ¿Existe una nueva solicitud de grabación?
                # =================================================

                record_request = None

                with self._record_lock:

                    if self._record_request is not None:

                        record_request = (
                            self._record_request
                        )

                        self._record_request = None

                if record_request is not None:

                    output_path, max_seconds = (
                        record_request
                    )

                    if self._video_writer is None:

                        self._start_recording(
                            frame=frame,
                            cap=cap,
                            output_path=output_path,
                            max_seconds=max_seconds,
                        )

                # =================================================
                # GRABAR FRAME ORIGINAL
                # =================================================

                if self._video_writer is not None:

                    self._update_recording(
                        frame
                    )


                # -------------------------------------------------
                # ¿Es momento de inferir?
                # -------------------------------------------------

                should_infer = False

                if inference_interval == 0:

                    should_infer = True

                elif (
                    current_time
                    - last_inference_time
                    >= inference_interval
                ):

                    should_infer = True

                # -------------------------------------------------
                # YOLO
                # -------------------------------------------------

                if should_infer:

                    inference_start = (
                        time.perf_counter()
                    )

                    with torch.inference_mode():

                        results = (
                            model.predict(
                                source=frame,
                                **prediction_kwargs,
                            )
                        )

                    inference_end = (
                        time.perf_counter()
                    )

                    inference_ms = (
                        inference_end
                        - inference_start
                    ) * 1000

                    if results:

                        self.cache_result(
                            results[0],
                            frame.shape,
                        )

                    # ---------------------------------------------
                    # Medir FPS real de inferencia
                    # ---------------------------------------------

                    if (
                        previous_inference_time
                        is not None
                    ):

                        delta = (
                            current_time
                            - previous_inference_time
                        )

                        if delta > 0:

                            measured_inference_fps = (
                                1.0 / delta
                            )

                    previous_inference_time = (
                        current_time
                    )

                    last_inference_time = (
                        current_time
                    )

                    # ---------------------------------------------
                    # Eliminar referencias al Result
                    # ---------------------------------------------

                    del results

                # -------------------------------------------------
                # Superponer última predicción
                # -------------------------------------------------

                display_frame = (
                    self.draw_cached_predictions(
                        frame
                    )
                )

                # -------------------------------------------------
                # Guardamos SOLO el último frame
                # -------------------------------------------------

                self.set_latest_frame(
                    display_frame
                )

                # -------------------------------------------------
                # FPS cámara
                # -------------------------------------------------

                elapsed = (
                    current_time
                    - fps_start
                )

                if elapsed >= 1.0:

                    camera_fps = (
                        fps_counter
                        / elapsed
                    )

                    fps_counter = 0

                    fps_start = (
                        current_time
                    )

                    self.metrics_updated.emit(
                        camera_fps,
                        measured_inference_fps,
                        inference_ms,
                    )

                # Pequeña pausa para no ocupar CPU
                # innecesariamente en el loop
                self.msleep(
                    1
                )

        except Exception as error:

            self.error_message.emit(
                f"{type(error).__name__}: "
                f"{error}"
            )

        finally:

            self.running = False


            # ---------------------------------------------
            # Si todavía estaba grabando, cerrar video
            # ---------------------------------------------

            if self._video_writer is not None:

                self._finish_recording()
                
            # -------------------------------------------------
            # Liberar cámara
            # -------------------------------------------------

            if cap is not None:

                cap.release()

            # -------------------------------------------------
            # Liberar modelo
            # -------------------------------------------------

            if model is not None:

                del model

            self.cached_boxes = []

            self.cached_masks = []

            self.cached_classification = None

            with self._frame_lock:

                self._latest_frame = None

            gc.collect()

            # Limpiar CUDA UNA SOLA VEZ
            # al finalizar, no en cada frame
            if torch.cuda.is_available():

                try:

                    torch.cuda.empty_cache()

                except Exception:

                    pass

            self.realtime_stopped.emit()

    # =========================================================
    # CONTROL DE GRABACIÓN
    # =========================================================

    def request_recording(
        self,
        output_path,
        max_seconds=30.0,
    ):

        with self._record_lock:

            self._record_request = (
                str(output_path),
                float(max_seconds),
            )

            self._record_stop_requested = False


    def request_stop_recording(self):

        with self._record_lock:

            self._record_stop_requested = True


    def is_recording(self):

        return (
            self._video_writer
            is not None
        )

    def _start_recording(
        self,
        frame,
        cap,
        output_path,
        max_seconds,
    ):

        try:

            height, width = (
                frame.shape[:2]
            )

            # ---------------------------------------------
            # FPS de la cámara
            # ---------------------------------------------

            camera_fps = float(
                cap.get(
                    cv2.CAP_PROP_FPS
                )
                or 0
            )

            # Muchas cámaras devuelven -1 o 0
            if (
                camera_fps <= 1
                or camera_fps > 120
            ):

                camera_fps = 30.0

            # ---------------------------------------------
            # Carpeta
            # ---------------------------------------------

            from pathlib import Path

            output_path = Path(
                output_path
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            # ---------------------------------------------
            # Codec MP4
            # ---------------------------------------------

            fourcc = (
                cv2.VideoWriter_fourcc(
                    *"mp4v"
                )
            )

            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                camera_fps,
                (
                    width,
                    height,
                ),
            )

            if not writer.isOpened():

                raise RuntimeError(
                    "No fue posible crear "
                    "el archivo de video."
                )

            self._video_writer = writer

            self._recording_path = str(
                output_path
            )

            self._recording_start_time = (
                time.perf_counter()
            )

            self._recording_max_seconds = min(
                float(max_seconds),
                30.0,
            )

            self._last_recording_signal_time = 0

            self.recording_started.emit(
                self._recording_path
            )

        except Exception as error:

            self.recording_error.emit(
                f"{type(error).__name__}: "
                f"{error}"
            )


    def _update_recording(
        self,
        frame,
    ):

        if self._video_writer is None:

            return

        # ---------------------------------------------
        # Guardar frame ORIGINAL
        # ---------------------------------------------

        self._video_writer.write(
            frame
        )

        current_time = (
            time.perf_counter()
        )

        elapsed = (
            current_time
            - self._recording_start_time
        )

        # ---------------------------------------------
        # Actualizar contador cada ~0.1 s
        # ---------------------------------------------

        if (
            current_time
            - self._last_recording_signal_time
            >= 0.1
        ):

            self.recording_time_updated.emit(
                elapsed
            )

            self._last_recording_signal_time = (
                current_time
            )

        # ---------------------------------------------
        # ¿Usuario pidió detener?
        # ---------------------------------------------

        with self._record_lock:

            stop_requested = (
                self._record_stop_requested
            )

        # ---------------------------------------------
        # Máximo 30 segundos
        # ---------------------------------------------

        if (
            stop_requested
            or elapsed
            >= self._recording_max_seconds
        ):

            self._finish_recording()

    def _finish_recording(self):

        if self._video_writer is None:

            return

        elapsed = (
            time.perf_counter()
            - self._recording_start_time
        )

        elapsed = min(
            elapsed,
            self._recording_max_seconds,
        )

        try:

            self._video_writer.release()

        except Exception:

            pass

        self._video_writer = None

        path = self._recording_path

        self._recording_path = None

        self._recording_start_time = None

        with self._record_lock:

            self._record_stop_requested = False

        self.recording_time_updated.emit(
            elapsed
        )

        self.recording_finished.emit(
            path,
            elapsed,
        )







