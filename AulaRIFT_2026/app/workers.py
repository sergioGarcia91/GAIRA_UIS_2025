# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 19:38:11 2026

@author: thesa
"""

from PySide6.QtCore import QThread, Signal

from app.model_manager import validate_model

from app.camera_manager import detect_cameras


from app.image_predictor import (
    process_image_with_model,
)

from app.video_predictor import (
    process_video_with_model,
)

class ModelValidationThread(QThread):

    model_checked = Signal(object)

    progress_changed = Signal(
        int,
        int,
    )

    validation_finished = Signal()

    status_message = Signal(str)


    def __init__(
        self,
        model_paths,
        root_folder,
        parent=None,
    ):

        super().__init__(parent)

        self.model_paths = model_paths

        self.root_folder = root_folder


    def run(self):

        total = len(
            self.model_paths
        )

        for number, model_path in enumerate(
            self.model_paths,
            start=1,
        ):

            if self.isInterruptionRequested():
                break

            self.status_message.emit(
                f"Validando {model_path.name}"
            )

            info = validate_model(
                model_path=model_path,
                root_folder=self.root_folder,
                run_test_inference=True,
            )

            self.model_checked.emit(info)

            self.progress_changed.emit(
                number,
                total,
            )

        self.validation_finished.emit()


class CameraScanThread(QThread):

    cameras_found = Signal(object)

    scan_finished = Signal()

    status_message = Signal(str)


    def __init__(
        self,
        max_index=9,
        parent=None,
    ):

        super().__init__(parent)

        self.max_index = max_index


    def run(self):

        self.status_message.emit(
            "Buscando cámaras..."
        )

        cameras = detect_cameras(
            max_index=self.max_index
        )

        self.cameras_found.emit(
            cameras
        )

        self.scan_finished.emit()

class ImagePredictionThread(QThread):

    progress_changed = Signal(
        int,
        int,
    )

    status_message = Signal(str)

    model_finished = Signal(object)

    prediction_finished = Signal()

    prediction_error = Signal(str)


    def __init__(
        self,
        models,
        image_path,
        output_folder,
        settings,
        parent=None,
    ):

        super().__init__(
            parent
        )

        self.models = models

        self.image_path = image_path

        self.output_folder = (
            output_folder
        )

        self.settings = settings


    def run(self):

        total = len(
            self.models
        )

        try:

            for number, model_info in enumerate(
                self.models,
                start=1,
            ):

                if self.isInterruptionRequested():
                    break

                self.status_message.emit(
                    f"Procesando "
                    f"{model_info.relative_name}"
                )

                result = (
                    process_image_with_model(
                        model_path=model_info.path,
                        image_path=self.image_path,
                        output_folder=self.output_folder,
                        settings=self.settings,
                    )
                )

                self.model_finished.emit(
                    result
                )

                self.progress_changed.emit(
                    number,
                    total,
                )

            self.prediction_finished.emit()

        except Exception as error:

            self.prediction_error.emit(
                f"{type(error).__name__}: "
                f"{error}"
            )


class VideoPredictionThread(QThread):

    model_started = Signal(
        str,
        int,
        int,
    )

    model_progress = Signal(
        int,
        int,
    )

    model_finished = Signal(object)

    processing_finished = Signal()

    processing_error = Signal(str)

    status_message = Signal(str)


    def __init__(
        self,
        models,
        video_path,
        output_folder,
        settings,
        inference_fps=1,
        parent=None,
    ):

        super().__init__(
            parent
        )

        self.models = models

        self.video_path = video_path

        self.output_folder = (
            output_folder
        )

        self.settings = settings

        self.inference_fps = (
            inference_fps
        )


    def run(self):

        total_models = len(
            self.models
        )

        try:

            for model_number, model_info in enumerate(
                self.models,
                start=1,
            ):

                if self.isInterruptionRequested():

                    break

                self.model_started.emit(
                    model_info.relative_name,
                    model_number,
                    total_models,
                )

                self.status_message.emit(
                    f"Procesando video con "
                    f"{model_info.relative_name}"
                )

                result = (
                    process_video_with_model(
                        model_path=model_info.path,
                        model_label=model_info.relative_name,
                        video_path=self.video_path,
                        output_root=self.output_folder,
                        settings=self.settings,
                        inference_fps=self.inference_fps,
                        progress_callback=(
                            self._emit_progress
                        ),
                        interruption_callback=(
                            self.isInterruptionRequested
                        ),
                    )
                )

                self.model_finished.emit(
                    result
                )

            self.processing_finished.emit()

        except Exception as error:

            self.processing_error.emit(
                f"{type(error).__name__}: "
                f"{error}"
            )


    def _emit_progress(
        self,
        current,
        total,
    ):

        self.model_progress.emit(
            current,
            total,
        )


