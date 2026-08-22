# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 19:38:31 2026

@author: thesa
"""

from pathlib import Path

import torch

import shutil

import cv2

from datetime import datetime

from PySide6.QtCore import (
    Qt,
    QTimer,
)

from PySide6.QtGui import (
    QColor,
    QPixmap,
    QImage,
)

from app.realtime_predictor import (
    RealtimePredictionThread,
)

from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFrame,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QTabWidget,
    QSizePolicy,
)

from app.model_manager import (
    find_pt_models,
)

from app.settings import (
    YoloSettings,
)

from app.workers import (
    CameraScanThread,
    ModelValidationThread,
    ImagePredictionThread,
    VideoPredictionThread,
)

class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "YOLO Multi-Model Predictor — v0.1"
        )

        self.resize(
            590,#1180,
            380,#760,
        )

        # -----------------------------
        # Variables
        # -----------------------------

        self.settings = YoloSettings()

        self.valid_models = []

        self.invalid_models = []

        self.validation_thread = None

        self.camera_thread = None
        
        #self.image_thread = None

        #self.selected_image = None

        self.image_results_folder = None
        
        # Tiempo real
        self.realtime_thread = None
        
        self.realtime_display_timer = QTimer(
            self
        )
        
        # Actualizamos pantalla a unos 15 FPS.
        # Esto NO significa que YOLO funcione a 15 FPS.
        self.realtime_display_timer.setInterval(
            66
        )
        
        self.realtime_display_timer.timeout.connect(
            self.update_realtime_preview
        )

        self.video_thread = None
        
        self.pending_video_path = None
        
        self.pending_video_folder = None
        
        self.pending_video_settings = None
        
        self.pending_video_inference_fps = 1
        
        self.recording_active = False

        # -----------------------------
        # Crear interfaz
        # -----------------------------

        self.build_ui()

        self.populate_devices()

        self.connect_signals()

        self.statusBar().showMessage(
            "Listo"
        )
        



    # =========================================================
    # INTERFAZ
    # =========================================================

    def build_ui(self):
    
        # =========================================================
        # VENTANA CENTRAL
        # =========================================================
    
        central = QWidget()
    
        self.setCentralWidget(
            central
        )
    
        main_layout = QVBoxLayout(
            central
        )
    
        main_layout.setContentsMargins(
            10,
            10,
            10,
            10,
        )
    
        main_layout.setSpacing(
            8
        )
    
        # =========================================================
        # TÍTULO
        # =========================================================
    
        title = QLabel(
            "YOLO Multi-Model Predictor - Aula RIFT"
        )
    
        title.setAlignment(
            Qt.AlignCenter
        )
    
        title.setStyleSheet(
            """
            font-size: 24px;
            font-weight: bold;
            """
        )
    
        main_layout.addWidget(
            title
        )
    
        subtitle = QLabel(
            "Aplicación para validación y predicción "
            "con múltiples modelos YOLO"
        )
    
        subtitle.setAlignment(
            Qt.AlignCenter
        )
    
        subtitle.setStyleSheet(
            "color: #666666;"
        )
    
        main_layout.addWidget(
            subtitle
        )
    
        # =========================================================
        # PESTAÑAS PRINCIPALES
        # =========================================================
    
        self.main_tabs = QTabWidget()
    
        main_layout.addWidget(
            self.main_tabs,
            stretch=1,
        )
    
        # #########################################################
        #
        # TAB 1
        # CONFIGURACIÓN
        #
        # #########################################################
    
        config_tab = QWidget()
    
        config_tab_layout = QVBoxLayout(
            config_tab
        )
    
        config_tab_layout.setContentsMargins(
            10,
            10,
            10,
            10,
        )
    
        config_tab_layout.setSpacing(
            10
        )
    
        # =========================================================
        # 1. CARPETAS
        # =========================================================
    
        folders_group = QGroupBox(
            "1. Configuración de archivos"
        )
    
        folders_layout = QGridLayout(
            folders_group
        )
    
        # ---------------------------------------------------------
        # Carpeta modelos
        # ---------------------------------------------------------
    
        self.models_path_edit = QLineEdit()
    
        self.models_path_edit.setPlaceholderText(
            "Seleccione la carpeta que contiene los modelos .pt"
        )
    
        self.models_button = QPushButton(
            "Examinar"
        )
    
        self.validate_button = QPushButton(
            "Buscar y validar modelos"
        )
    
        folders_layout.addWidget(
            QLabel("Carpeta de modelos:"),
            0,
            0,
        )
    
        folders_layout.addWidget(
            self.models_path_edit,
            0,
            1,
        )
    
        folders_layout.addWidget(
            self.models_button,
            0,
            2,
        )
    
        folders_layout.addWidget(
            self.validate_button,
            0,
            3,
        )
    
        # ---------------------------------------------------------
        # Carpeta salida
        # ---------------------------------------------------------
    
        self.output_path_edit = QLineEdit()
    
        self.output_path_edit.setPlaceholderText(
            "Seleccione la carpeta donde se guardarán los resultados"
        )
    
        self.output_button = QPushButton(
            "Examinar"
        )
    
        folders_layout.addWidget(
            QLabel("Carpeta de salida:"),
            1,
            0,
        )
    
        folders_layout.addWidget(
            self.output_path_edit,
            1,
            1,
            1,
            2,
        )
    
        folders_layout.addWidget(
            self.output_button,
            1,
            3,
        )
    
        config_tab_layout.addWidget(
            folders_group
        )
    
        # =========================================================
        # 2. MODELOS YOLO
        # =========================================================
    
        models_group = QGroupBox(
            "2. Validación de modelos YOLO"
        )
    
        models_layout = QVBoxLayout(
            models_group
        )
    
        # ---------------------------------------------------------
        # Contadores
        # ---------------------------------------------------------
    
        counters_layout = QHBoxLayout()
    
        self.total_label = QLabel(
            "Encontrados: 0"
        )
    
        self.valid_label = QLabel(
            "Válidos: 0"
        )
    
        self.invalid_label = QLabel(
            "Con error: 0"
        )
    
        counters_layout.addWidget(
            self.total_label
        )
    
        counters_layout.addSpacing(
            20
        )
    
        counters_layout.addWidget(
            self.valid_label
        )
    
        counters_layout.addSpacing(
            20
        )
    
        counters_layout.addWidget(
            self.invalid_label
        )
    
        counters_layout.addStretch()
    
        models_layout.addLayout(
            counters_layout
        )
    
        # ---------------------------------------------------------
        # Progreso
        # ---------------------------------------------------------
    
        self.model_progress = QProgressBar()
    
        self.model_progress.setValue(
            0
        )
    
        models_layout.addWidget(
            self.model_progress
        )
    
        # ---------------------------------------------------------
        # Tabla
        # ---------------------------------------------------------
    
        self.models_table = QTableWidget(
            0,
            6,
        )
    
        self.models_table.setHorizontalHeaderLabels(
            [
                "Modelo",
                "Estado",
                "Tarea",
                "N.º clases",
                "Clases",
                "Observación",
            ]
        )
    
        self.models_table.setEditTriggers(
            QAbstractItemView.NoEditTriggers
        )
    
        self.models_table.setSelectionBehavior(
            QAbstractItemView.SelectRows
        )
    
        self.models_table.setAlternatingRowColors(
            True
        )
    
        self.models_table.verticalHeader().setVisible(
            False
        )
    
        header = (
            self.models_table.horizontalHeader()
        )
    
        header.setSectionResizeMode(
            0,
            QHeaderView.Stretch,
        )
    
        header.setSectionResizeMode(
            1,
            QHeaderView.ResizeToContents,
        )
    
        header.setSectionResizeMode(
            2,
            QHeaderView.ResizeToContents,
        )
    
        header.setSectionResizeMode(
            3,
            QHeaderView.ResizeToContents,
        )
    
        header.setSectionResizeMode(
            4,
            QHeaderView.Stretch,
        )
    
        header.setSectionResizeMode(
            5,
            QHeaderView.Stretch,
        )
    
        models_layout.addWidget(
            self.models_table,
            stretch=1,
        )
    
        config_tab_layout.addWidget(
            models_group,
            stretch=1,
        )
    
        # ---------------------------------------------------------
        # Instrucción inferior
        # ---------------------------------------------------------
    
        config_info = QLabel(
            "Una vez validados los modelos, continúe a la pestaña "
            "'Predicción'. Solo los modelos válidos estarán disponibles."
        )
    
        config_info.setWordWrap(
            True
        )
    
        config_info.setStyleSheet(
            """
            padding: 8px;
            background-color: #f5f5f5;
            border: 1px solid #cccccc;
            border-radius: 4px;
            """
        )
    
        config_tab_layout.addWidget(
            config_info
        )
    
        self.main_tabs.addTab(
            config_tab,
            "1. Configuración",
        )
    
        # #########################################################
        #
        # TAB 2
        # PREDICCIÓN
        #
        # #########################################################
    
        prediction_tab = QWidget()
    
        prediction_tab_layout = QVBoxLayout(
            prediction_tab
        )
    
        prediction_tab_layout.setContentsMargins(
            10,
            10,
            10,
            10,
        )
    
        prediction_tab_layout.setSpacing(
            8
        )
    
        # =========================================================
        # PARÁMETROS COMUNES
        # =========================================================
    
        parameters_group = QGroupBox(
            "Parámetros de inferencia"
        )
    
        parameters_layout = QHBoxLayout(
            parameters_group
        )
    
        # ---------------------------------------------------------
        # Confianza
        # ---------------------------------------------------------
    
        self.conf_spin = QDoubleSpinBox()
    
        self.conf_spin.setRange(
            0,
            1,
        )
    
        self.conf_spin.setDecimals(
            2
        )
    
        self.conf_spin.setSingleStep(
            0.05
        )
    
        self.conf_spin.setValue(
            self.settings.conf
        )
    
        parameters_layout.addWidget(
            QLabel("Confianza:")
        )
    
        parameters_layout.addWidget(
            self.conf_spin
        )
    
        # ---------------------------------------------------------
        # IoU
        # ---------------------------------------------------------
    
        self.iou_spin = QDoubleSpinBox()
    
        self.iou_spin.setRange(
            0,
            1,
        )
    
        self.iou_spin.setDecimals(
            2
        )
    
        self.iou_spin.setSingleStep(
            0.05
        )
    
        self.iou_spin.setValue(
            self.settings.iou
        )
    
        parameters_layout.addWidget(
            QLabel("IoU:")
        )
    
        parameters_layout.addWidget(
            self.iou_spin
        )
    
        # ---------------------------------------------------------
        # ImgSz
        # ---------------------------------------------------------
    
        self.imgsz_combo = QComboBox()
    
        self.imgsz_combo.setEditable(
            True
        )
    
        for value in [
            320,
            416,
            512,
            640,
            768,
            960,
            1024,
            1280,
        ]:
    
            self.imgsz_combo.addItem(
                str(value)
            )
    
        self.imgsz_combo.setCurrentText(
            str(self.settings.imgsz)
        )
    
        parameters_layout.addWidget(
            QLabel("imgsz:")
        )
    
        parameters_layout.addWidget(
            self.imgsz_combo
        )
    
        # ---------------------------------------------------------
        # Máximo de detecciones
        # ---------------------------------------------------------
    
        self.max_det_spin = QSpinBox()
    
        self.max_det_spin.setRange(
            1,
            10000,
        )
    
        self.max_det_spin.setValue(
            self.settings.max_det
        )
    
        parameters_layout.addWidget(
            QLabel("Máx. detecciones:")
        )
    
        parameters_layout.addWidget(
            self.max_det_spin
        )
    
        # ---------------------------------------------------------
        # Device
        # ---------------------------------------------------------
    
        self.device_combo = QComboBox()
    
        parameters_layout.addWidget(
            QLabel("Dispositivo:")
        )
    
        parameters_layout.addWidget(
            self.device_combo
        )
    
        # ---------------------------------------------------------
        # Frecuencia de inferencia
        # ---------------------------------------------------------
    
        self.inference_fps_combo = QComboBox()
    
        self.inference_fps_combo.addItem(
            "1 FPS",
            1,
        )
    
        self.inference_fps_combo.addItem(
            "2 FPS",
            2,
        )
    
        self.inference_fps_combo.addItem(
            "5 FPS",
            5,
        )
    
        self.inference_fps_combo.addItem(
            "10 FPS",
            10,
        )
    
        self.inference_fps_combo.addItem(
            "Todos",
            0,
        )
    
        parameters_layout.addWidget(
            QLabel("Inferencia:")
        )
    
        parameters_layout.addWidget(
            self.inference_fps_combo
        )
    
        # ---------------------------------------------------------
        # FP16
        # ---------------------------------------------------------
    
        self.half_checkbox = QCheckBox(
            "FP16"
        )
    
        parameters_layout.addWidget(
            self.half_checkbox
        )
    
        self.restore_button = QPushButton(
            "Restaurar"
        )
    
        parameters_layout.addWidget(
            self.restore_button
        )
    
        prediction_tab_layout.addWidget(
            parameters_group
        )
    
        # =========================================================
        # SUBPESTAÑAS
        # =========================================================
    
        self.prediction_mode_tabs = QTabWidget()
    
        prediction_tab_layout.addWidget(
            self.prediction_mode_tabs,
            stretch=1,
        )
    
        # #########################################################
        #
        # SUBTAB IMAGEN
        #
        # #########################################################
    
        image_tab = QWidget()
    
        image_tab_layout = QHBoxLayout(
            image_tab
        )
    
        image_tab_layout.setContentsMargins(
            8,
            8,
            8,
            8,
        )
    
        # =========================================================
        # Imagen - zona izquierda
        # =========================================================
    
        image_left = QVBoxLayout()
    
        image_file_layout = QHBoxLayout()
    
        self.image_path_edit = QLineEdit()
    
        self.image_path_edit.setReadOnly(
            True
        )
    
        self.image_path_edit.setPlaceholderText(
            "Seleccione una imagen..."
        )
    
        self.select_image_button = QPushButton(
            "Seleccionar imagen"
        )
    
        image_file_layout.addWidget(
            self.image_path_edit
        )
    
        image_file_layout.addWidget(
            self.select_image_button
        )
    
        image_left.addLayout(
            image_file_layout
        )
    
        self.image_preview = QLabel(
            "Seleccione una imagen"
        )
    
        self.image_preview.setAlignment(
            Qt.AlignCenter
        )
    
        self.image_preview.setMinimumSize(
            640,
            420,
        )
    
        self.image_preview.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
    
        self.image_preview.setFrameShape(
            QFrame.Box
        )
    
        self.image_preview.setStyleSheet(
            """
            background-color: #202020;
            color: #dddddd;
            border: 1px solid #555555;
            """
        )
    
        image_left.addWidget(
            self.image_preview,
            stretch=1,
        )
    
        image_tab_layout.addLayout(
            image_left,
            stretch=4,
        )
    
        # =========================================================
        # Imagen - controles derechos
        # =========================================================
    
        image_right = QVBoxLayout()
    
        self.predict_image_button = QPushButton(
            "Ejecutar todos los modelos"
        )
    
        self.predict_image_button.setEnabled(
            False
        )
    
        image_right.addWidget(
            self.predict_image_button
        )
    
        self.image_progress = QProgressBar()
    
        self.image_progress.setValue(
            0
        )
    
        image_right.addWidget(
            self.image_progress
        )
    
        self.current_model_label = QLabel(
            "Modelo actual: —"
        )
    
        self.current_model_label.setWordWrap(
            True
        )
    
        image_right.addWidget(
            self.current_model_label
        )
    
        self.image_result_label = QLabel(
            "Resultados: —"
        )
    
        self.image_result_label.setWordWrap(
            True
        )
    
        image_right.addWidget(
            self.image_result_label
        )
    
        image_right.addStretch()
    
        image_tab_layout.addLayout(
            image_right,
            stretch=1,
        )
    
        self.prediction_mode_tabs.addTab(
            image_tab,
            "Predicción en imagen",
        )
    
        # #########################################################
        #
        # SUBTAB TIEMPO REAL
        #
        # #########################################################
    
        realtime_tab = QWidget()
    
        realtime_tab_layout = QHBoxLayout(
            realtime_tab
        )
    
        realtime_tab_layout.setContentsMargins(
            8,
            8,
            8,
            8,
        )
    
        # =========================================================
        # Video
        # =========================================================
    
        realtime_left = QVBoxLayout()
    
        self.realtime_preview = QLabel(
            "Cámara detenida"
        )
    
        self.realtime_preview.setAlignment(
            Qt.AlignCenter
        )
    
        self.realtime_preview.setMinimumSize(
            640,
            420,
        )
    
        self.realtime_preview.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
    
        self.realtime_preview.setFrameShape(
            QFrame.Box
        )
    
        self.realtime_preview.setStyleSheet(
            """
            background-color: #202020;
            color: #dddddd;
            border: 1px solid #555555;
            """
        )
    
        realtime_left.addWidget(
            self.realtime_preview,
            stretch=1,
        )
    
        realtime_tab_layout.addLayout(
            realtime_left,
            stretch=4,
        )
    
        # =========================================================
        # Controles tiempo real
        # =========================================================
    
        realtime_right = QVBoxLayout()
    
        # ---------------------------------------------------------
        # Cámara
        # ---------------------------------------------------------
    
        camera_group = QGroupBox(
            "Cámara"
        )
    
        camera_layout = QVBoxLayout(
            camera_group
        )
    
        self.camera_combo = QComboBox()
    
        self.camera_combo.addItem(
            "Aún no se han buscado cámaras",
            None,
        )
    
        self.refresh_camera_button = QPushButton(
            "Actualizar cámaras"
        )
    
        self.camera_detail_label = QLabel(
            "Puerto: —\n"
            "Backend: —"
        )
    
        camera_layout.addWidget(
            self.camera_combo
        )
    
        camera_layout.addWidget(
            self.refresh_camera_button
        )
    
        camera_layout.addWidget(
            self.camera_detail_label
        )
    
        realtime_right.addWidget(
            camera_group
        )
    
        # ---------------------------------------------------------
        # Modelo
        # ---------------------------------------------------------
    
        realtime_right.addWidget(
            QLabel(
                "Modelo para tiempo real:"
            )
        )
    
        self.realtime_model_combo = QComboBox()
    
        self.realtime_model_combo.addItem(
            "Valide primero los modelos",
            None,
        )
    
        realtime_right.addWidget(
            self.realtime_model_combo
        )
    
        # ---------------------------------------------------------
        # Iniciar / detener
        # ---------------------------------------------------------
    
        realtime_buttons = QHBoxLayout()
    
        self.start_realtime_button = QPushButton(
            "▶ Iniciar"
        )
    
        self.stop_realtime_button = QPushButton(
            "■ Detener"
        )
    
        self.start_realtime_button.setEnabled(
            False
        )
    
        self.stop_realtime_button.setEnabled(
            False
        )
    
        realtime_buttons.addWidget(
            self.start_realtime_button
        )
    
        realtime_buttons.addWidget(
            self.stop_realtime_button
        )
    
        realtime_right.addLayout(
            realtime_buttons
        )
    
        # ---------------------------------------------------------
        # Grabación
        # ---------------------------------------------------------
    
        record_buttons = QHBoxLayout()
    
        self.record_button = QPushButton(
            "● Grabar"
        )
    
        self.stop_record_button = QPushButton(
            "■ Detener grabación"
        )
    
        self.record_button.setEnabled(
            False
        )
    
        self.stop_record_button.setEnabled(
            False
        )
    
        record_buttons.addWidget(
            self.record_button
        )
    
        record_buttons.addWidget(
            self.stop_record_button
        )
    
        realtime_right.addLayout(
            record_buttons
        )
    
        self.recording_time_label = QLabel(
            "Grabación: 00:00 / 00:30"
        )
    
        realtime_right.addWidget(
            self.recording_time_label
        )
    
        # ---------------------------------------------------------
        # Procesamiento
        # ---------------------------------------------------------
    
        self.video_models_label = QLabel(
            "Procesamiento: —"
        )
    
        realtime_right.addWidget(
            self.video_models_label
        )
    
        self.video_models_progress = QProgressBar()
    
        self.video_models_progress.setValue(
            0
        )
    
        realtime_right.addWidget(
            self.video_models_progress
        )
    
        self.video_frame_progress = QProgressBar()
    
        self.video_frame_progress.setValue(
            0
        )
    
        realtime_right.addWidget(
            self.video_frame_progress
        )
    
        # ---------------------------------------------------------
        # Métricas
        # ---------------------------------------------------------
    
        realtime_right.addSpacing(
            10
        )
    
        self.camera_fps_label = QLabel(
            "FPS cámara: —"
        )
    
        self.yolo_fps_label = QLabel(
            "FPS inferencia: —"
        )
    
        self.inference_time_label = QLabel(
            "Tiempo YOLO: —"
        )
    
        self.realtime_status_label = QLabel(
            "Estado: detenido"
        )
    
        self.realtime_status_label.setWordWrap(
            True
        )
    
        realtime_right.addWidget(
            self.camera_fps_label
        )
    
        realtime_right.addWidget(
            self.yolo_fps_label
        )
    
        realtime_right.addWidget(
            self.inference_time_label
        )
    
        realtime_right.addWidget(
            self.realtime_status_label
        )
    
        realtime_right.addStretch()
    
        realtime_tab_layout.addLayout(
            realtime_right,
            stretch=1,
        )
    
        self.prediction_mode_tabs.addTab(
            realtime_tab,
            "Tiempo real",
        )
    
        # =========================================================
        # AGREGAR TAB PRINCIPAL
        # =========================================================
    
        self.main_tabs.addTab(
            prediction_tab,
            "2. Predicción",
        )
    
        # =========================================================
        # STATUS
        # =========================================================
    
        self.setStatusBar(
            QStatusBar()
        )





    # =========================================================
    # SEÑALES
    # =========================================================

    def connect_signals(self):
        
        self.select_image_button.clicked.connect(
            self.select_image
        )

        self.predict_image_button.clicked.connect(
            self.start_image_prediction
        )

        self.models_button.clicked.connect(
            self.select_models_folder
        )

        self.output_button.clicked.connect(
            self.select_output_folder
        )

        self.validate_button.clicked.connect(
            self.start_model_validation
        )

        self.refresh_camera_button.clicked.connect(
            self.start_camera_scan
        )

        self.camera_combo.currentIndexChanged.connect(
            self.update_camera_details
        )

        self.restore_button.clicked.connect(
            self.restore_defaults
        )
        
        self.start_realtime_button.clicked.connect(
            self.start_realtime_prediction
        )
        
        self.stop_realtime_button.clicked.connect(
            self.stop_realtime_prediction
        )

        self.record_button.clicked.connect(
            self.start_recording
        )
        
        self.stop_record_button.clicked.connect(
            self.stop_recording
        )

    def refresh_realtime_models(self):
    
        self.realtime_model_combo.clear()
    
        if not self.valid_models:
    
            self.realtime_model_combo.addItem(
                "No hay modelos válidos",
                None,
            )
    
            self.start_realtime_button.setEnabled(
                False
            )
    
            return
    
        for model_info in self.valid_models:
    
            self.realtime_model_combo.addItem(
                model_info.relative_name,
                model_info,
            )
    
        self.start_realtime_button.setEnabled(
            True
        )

    # =========================================================
    # CARPETAS
    # =========================================================

    def select_models_folder(self):

        folder = QFileDialog.getExistingDirectory(
            self,
            "Seleccionar carpeta de modelos",
        )

        if folder:

            self.models_path_edit.setText(
                folder
            )


    def select_output_folder(self):

        folder = QFileDialog.getExistingDirectory(
            self,
            "Seleccionar carpeta de salida",
        )

        if folder:

            self.output_path_edit.setText(
                folder
            )


    # =========================================================
    # MODELOS
    # =========================================================

    def start_model_validation(self):

        folder = self.models_path_edit.text().strip()

        if not folder:

            QMessageBox.warning(
                self,
                "Atención",
                "Seleccione primero una carpeta de modelos.",
            )

            return

        root = Path(folder)

        model_paths = find_pt_models(
            root
        )

        if not model_paths:

            QMessageBox.information(
                self,
                "Modelos",
                "No se encontraron archivos .pt.",
            )

            return

        # -----------------------------
        # Reiniciar
        # -----------------------------

        self.valid_models = []

        self.invalid_models = []

        self.models_table.setRowCount(
            0
        )

        self.total_label.setText(
            f"Encontrados: {len(model_paths)}"
        )

        self.valid_label.setText(
            "Válidos: 0"
        )

        self.invalid_label.setText(
            "Con error: 0"
        )

        self.model_progress.setValue(
            0
        )

        # -----------------------------
        # Thread
        # -----------------------------

        self.validation_thread = ModelValidationThread(
            model_paths,
            root,
            self,
        )

        self.validation_thread.model_checked.connect(
            self.add_model_result
        )

        self.validation_thread.progress_changed.connect(
            self.update_progress
        )

        self.validation_thread.status_message.connect(
            self.statusBar().showMessage
        )

        self.validation_thread.validation_finished.connect(
            self.validation_finished
        )

        self.validation_thread.start()


    def add_model_result(
        self,
        info,
    ):

        if info.valid:

            self.valid_models.append(
                info
            )

        else:

            self.invalid_models.append(
                info
            )

        row = self.models_table.rowCount()

        self.models_table.insertRow(
            row
        )

        # -----------------------------
        # Modelo
        # -----------------------------

        model_item = QTableWidgetItem(
            info.relative_name
        )

        model_item.setToolTip(
            info.path
        )

        # -----------------------------
        # Estado
        # -----------------------------

        status_text = (
            "Válido"
            if info.valid
            else "Error"
        )

        status_item = QTableWidgetItem(
            status_text
        )

        if info.valid:

            status_item.setBackground(
                QColor(
                    220,
                    245,
                    225,
                )
            )

        else:

            status_item.setBackground(
                QColor(
                    255,
                    225,
                    225,
                )
            )

        # -----------------------------
        # Datos
        # -----------------------------

        task_item = QTableWidgetItem(
            info.task
        )

        class_count_item = QTableWidgetItem(
            str(info.class_count)
        )

        classes_item = QTableWidgetItem(
            info.classes_preview
        )

        observation = (
            "Carga e inferencia correctas"
            if info.valid
            else info.error
        )

        observation_item = QTableWidgetItem(
            observation
        )

        observation_item.setToolTip(
            observation
        )

        # -----------------------------
        # Tabla
        # -----------------------------

        self.models_table.setItem(
            row,
            0,
            model_item,
        )

        self.models_table.setItem(
            row,
            1,
            status_item,
        )

        self.models_table.setItem(
            row,
            2,
            task_item,
        )

        self.models_table.setItem(
            row,
            3,
            class_count_item,
        )

        self.models_table.setItem(
            row,
            4,
            classes_item,
        )

        self.models_table.setItem(
            row,
            5,
            observation_item,
        )

        # -----------------------------
        # Contadores
        # -----------------------------

        self.valid_label.setText(
            f"Válidos: {len(self.valid_models)}"
        )

        self.invalid_label.setText(
            f"Con error: {len(self.invalid_models)}"
        )


    def update_progress(
        self,
        current,
        total,
    ):

        if total == 0:
            return

        percent = int(
            current
            / total
            * 100
        )

        self.model_progress.setValue(
            percent
        )

        self.model_progress.setFormat(
            f"{current}/{total} — {percent}%"
        )


    def validation_finished(self):

        self.statusBar().showMessage(
            "Validación terminada"
        )
        self.refresh_realtime_models()
        
        if self.selected_image:
            self.predict_image_button.setEnabled(
                len(self.valid_models) > 0
            )

    def start_realtime_prediction(self):
    
        # -------------------------------------------------
        # Evitar iniciar dos veces
        # -------------------------------------------------
    
        if (
            self.realtime_thread is not None
            and self.realtime_thread.isRunning()
        ):
    
            return
    
        # -------------------------------------------------
        # Cámara
        # -------------------------------------------------
    
        camera = (
            self.camera_combo.currentData()
        )
    
        if camera is None:
    
            QMessageBox.warning(
                self,
                "Cámara",
                "Seleccione primero una cámara válida.",
            )
    
            return
    
        # -------------------------------------------------
        # Modelo
        # -------------------------------------------------
    
        model_info = (
            self.realtime_model_combo.currentData()
        )
    
        if model_info is None:
    
            QMessageBox.warning(
                self,
                "Modelo",
                "Seleccione un modelo válido.",
            )
    
            return
    
        # -------------------------------------------------
        # Parámetros YOLO
        # -------------------------------------------------
    
        settings = (
            self.get_yolo_settings()
        )
    
        inference_fps = (
            self.inference_fps_combo.currentData()
        )
    
        if inference_fps is None:
    
            inference_fps = 1
    
        # -------------------------------------------------
        # Crear thread
        # -------------------------------------------------
    
        self.realtime_thread = (
            RealtimePredictionThread(
                camera_index=camera.index,
                model_path=model_info.path,
                settings=settings,
                inference_fps=inference_fps,
                parent=self,
            )
        )
    
        # -------------------------------------------------
        # Señales
        # -------------------------------------------------
    
        self.realtime_thread.status_message.connect(
            self.update_realtime_status
        )
    
        self.realtime_thread.error_message.connect(
            self.realtime_error
        )
    
        self.realtime_thread.metrics_updated.connect(
            self.update_realtime_metrics
        )
    
        self.realtime_thread.realtime_started.connect(
            self.realtime_started
        )
    
        self.realtime_thread.realtime_stopped.connect(
            self.realtime_stopped
        )
    
        # -------------------------------------------------
        # Interfaz
        # -------------------------------------------------
    
        self.start_realtime_button.setEnabled(
            False
        )
    
        self.stop_realtime_button.setEnabled(
            True
        )
    
        self.realtime_model_combo.setEnabled(
            False
        )
    
        self.camera_combo.setEnabled(
            False
        )
    
        self.refresh_camera_button.setEnabled(
            False
        )
    
        self.realtime_status_label.setText(
            "Estado: iniciando..."
        )
    
        # -------------------------------------------------
        # Ejecutar
        # -------------------------------------------------
    
        self.realtime_thread.start()
    
        # Timer SOLO para consultar último frame
        self.realtime_display_timer.start()

    def update_realtime_preview(self):
    
        if self.realtime_thread is None:
    
            return
    
        if not self.realtime_thread.isRunning():
    
            return
    
        frame = (
            self.realtime_thread.get_latest_frame()
        )
    
        if frame is None:
    
            return
    
        # OpenCV usa BGR
        # Qt utiliza RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )
    
        height, width, channels = (
            rgb_frame.shape
        )
    
        bytes_per_line = (
            channels * width
        )
    
        qimage = QImage(
            rgb_frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        ).copy()
    
        pixmap = QPixmap.fromImage(
            qimage
        )
    
        pixmap = pixmap.scaled(
            self.realtime_preview.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
    
        self.realtime_preview.setPixmap(
            pixmap
        )

    def update_realtime_status(
        self,
        message,
    ):
    
        self.realtime_status_label.setText(
            f"Estado: {message}"
        )
    
        self.statusBar().showMessage(
            message
        )

    def update_realtime_metrics(
        self,
        camera_fps,
        inference_fps,
        inference_ms,
    ):
    
        self.camera_fps_label.setText(
            f"FPS cámara: "
            f"{camera_fps:.1f}"
        )
    
        self.yolo_fps_label.setText(
            f"FPS inferencia: "
            f"{inference_fps:.2f}"
        )
    
        self.inference_time_label.setText(
            f"Tiempo YOLO: "
            f"{inference_ms:.1f} ms"
        )
        
    # =========================================================
    # CÁMARAS
    # =========================================================

    def start_camera_scan(self):

        self.camera_combo.clear()

        self.camera_combo.addItem(
            "Buscando cámaras..."
        )

        self.refresh_camera_button.setEnabled(
            False
        )

        self.camera_thread = CameraScanThread(
            max_index=9,
            parent=self,
        )

        self.camera_thread.cameras_found.connect(
            self.populate_cameras
        )

        self.camera_thread.scan_finished.connect(
            self.camera_scan_finished
        )

        self.camera_thread.start()


    def populate_cameras(
        self,
        cameras,
    ):

        self.camera_combo.clear()

        if not cameras:

            self.camera_combo.addItem(
                "No se detectaron cámaras",
                None,
            )

            return

        for camera in cameras:

            self.camera_combo.addItem(
                camera.display_name,
                camera,
            )


    def camera_scan_finished(self):

        self.refresh_camera_button.setEnabled(
            True
        )

        self.statusBar().showMessage(
            "Búsqueda de cámaras terminada"
        )


    def update_camera_details(self):

        camera = self.camera_combo.currentData()

        if camera is None:

            self.camera_detail_label.setText(
                "Puerto: —\nBackend: —"
            )

            return

        self.camera_detail_label.setText(
            f"Puerto: {camera.index}\n"
            f"Backend: {camera.backend}\n"
            f"Resolución: "
            f"{camera.width} × {camera.height}\n"
            f"FPS: {camera.fps:.1f}"
        )


    # =========================================================
    # GPU / CPU
    # =========================================================

    def populate_devices(self):

        self.device_combo.clear()

        self.device_combo.addItem(
            "Automático",
            "auto",
        )

        self.device_combo.addItem(
            "CPU",
            "cpu",
        )

        if torch.cuda.is_available():

            gpu_count = (
                torch.cuda.device_count()
            )

            for index in range(
                gpu_count
            ):

                gpu_name = (
                    torch.cuda.get_device_name(
                        index
                    )
                )

                self.device_combo.addItem(
                    f"GPU {index} — {gpu_name}",
                    str(index),
                )


    # =========================================================
    # DEFAULTS
    # =========================================================

    def restore_defaults(self):

        defaults = YoloSettings()

        self.conf_spin.setValue(
            defaults.conf
        )

        self.iou_spin.setValue(
            defaults.iou
        )

        self.imgsz_combo.setCurrentText(
            str(defaults.imgsz)
        )

        self.max_det_spin.setValue(
            defaults.max_det
        )

        self.half_checkbox.setChecked(
            defaults.half
        )

        self.device_combo.setCurrentIndex(
            0
        )

        self.statusBar().showMessage(
            "Parámetros restaurados"
        )
        
        
    def select_image(self):
    
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar imagen",
            "",
            (
                "Imágenes "
                "(*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"
            ),
        )
    
        if not file_path:
            return
    
        self.selected_image = file_path
    
        self.image_path_edit.setText(
            file_path
        )
    
        # -------------------------------------------------
        # Mostrar vista previa
        # -------------------------------------------------
    
        pixmap = QPixmap(
            file_path
        )
    
        if pixmap.isNull():
    
            QMessageBox.warning(
                self,
                "Imagen",
                "No fue posible cargar la imagen.",
            )
    
            return
    
        scaled = pixmap.scaled(
            self.image_preview.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
    
        self.image_preview.setPixmap(
            scaled
        )
    
        # Habilitar predicción solamente
        # si existen modelos válidos
        self.predict_image_button.setEnabled(
            len(self.valid_models) > 0
        )
    
        self.statusBar().showMessage(
            "Imagen cargada"
        )
        
        
    def get_yolo_settings(self):
    
        try:
    
            imgsz = int(
                self.imgsz_combo.currentText()
            )
    
        except ValueError:
    
            imgsz = 640
    
        return {
    
            "conf": (
                self.conf_spin.value()
            ),
    
            "iou": (
                self.iou_spin.value()
            ),
    
            "imgsz": imgsz,
    
            "max_det": (
                self.max_det_spin.value()
            ),
    
            "device": (
                self.device_combo.currentData()
                or "auto"
            ),
    
            "half": (
                self.half_checkbox.isChecked()
            ),
        }

    def start_image_prediction(self):
    
        # -------------------------------------------------
        # Comprobar imagen
        # -------------------------------------------------
    
        if not self.selected_image:
    
            QMessageBox.warning(
                self,
                "Imagen",
                "Seleccione primero una imagen.",
            )
    
            return
    
        # -------------------------------------------------
        # Comprobar modelos
        # -------------------------------------------------
    
        if not self.valid_models:
    
            QMessageBox.warning(
                self,
                "Modelos",
                "No existen modelos válidos.",
            )
    
            return
    
        # -------------------------------------------------
        # Comprobar carpeta de salida
        # -------------------------------------------------
    
        output_root = (
            self.output_path_edit.text().strip()
        )
    
        if not output_root:
    
            QMessageBox.warning(
                self,
                "Carpeta de salida",
                "Seleccione primero la carpeta de salida.",
            )
    
            return
    
        output_root = Path(
            output_root
        )
    
        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )
    
        # -------------------------------------------------
        # Crear carpeta de esta ejecución
        # -------------------------------------------------
    
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    
        image_name = Path(
            self.selected_image
        ).stem
    
        execution_folder = (
            output_root
            / "Imagen"
            / f"{image_name}_{timestamp}"
        )
    
        execution_folder.mkdir(
            parents=True,
            exist_ok=True,
        )
    
        self.image_results_folder = (
            execution_folder
        )
    
        # -------------------------------------------------
        # Copiar original
        # -------------------------------------------------
    
        extension = Path(
            self.selected_image
        ).suffix
    
        original_copy = (
            execution_folder
            / f"original{extension}"
        )
    
        shutil.copy2(
            self.selected_image,
            original_copy,
        )
    
        # -------------------------------------------------
        # Preparar interfaz
        # -------------------------------------------------
    
        self.image_progress.setValue(
            0
        )
    
        self.current_model_label.setText(
            "Preparando predicción..."
        )
    
        self.image_result_label.setText(
            f"Resultados:\n{execution_folder}"
        )
    
        self.predict_image_button.setEnabled(
            False
        )
    
        # -------------------------------------------------
        # Parámetros
        # -------------------------------------------------
    
        settings = (
            self.get_yolo_settings()
        )
    
        # -------------------------------------------------
        # Thread
        # -------------------------------------------------
    
        self.image_thread = (
            ImagePredictionThread(
                models=self.valid_models,
                image_path=self.selected_image,
                output_folder=execution_folder,
                settings=settings,
                parent=self,
            )
        )
    
        self.image_thread.progress_changed.connect(
            self.update_image_progress
        )
    
        self.image_thread.status_message.connect(
            self.update_image_status
        )
    
        self.image_thread.model_finished.connect(
            self.image_model_finished
        )
    
        self.image_thread.prediction_finished.connect(
            self.image_prediction_finished
        )
    
        self.image_thread.prediction_error.connect(
            self.image_prediction_error
        )
    
        self.image_thread.start()
        
        
    def update_image_progress(
        self,
        current,
        total,
    ):
    
        if total == 0:
            return
    
        percent = int(
            current
            / total
            * 100
        )
    
        self.image_progress.setValue(
            percent
        )
    
        self.image_progress.setFormat(
            f"{current}/{total} — {percent}%"
        )

    def update_image_status(
        self,
        message,
    ):
    
        self.current_model_label.setText(
            message
        )
    
        self.statusBar().showMessage(
            message
        )


    def image_model_finished(
        self,
        result,
    ):
    
        model_name = result[
            "model"
        ]
    
        detections = result[
            "detections"
        ]
    
        self.current_model_label.setText(
            f"{model_name}: "
            f"{detections} detección(es)"
        )

    def image_prediction_finished(self):
    
        self.predict_image_button.setEnabled(
            True
        )
    
        self.current_model_label.setText(
            "Predicción terminada"
        )
    
        self.image_progress.setValue(
            100
        )
    
        self.statusBar().showMessage(
            "Predicción de imagen terminada"
        )
    
        QMessageBox.information(
            self,
            "Predicción terminada",
            (
                "Todos los modelos fueron procesados.\n\n"
                "Los resultados se guardaron en:\n"
                f"{self.image_results_folder}"
            ),
        )

    def image_prediction_error(
        self,
        message,
    ):
    
        self.predict_image_button.setEnabled(
            True
        )
    
        self.current_model_label.setText(
            "Error"
        )
    
        QMessageBox.critical(
            self,
            "Error en la predicción",
            message,
        )

    def realtime_started(self):
    
        self.realtime_status_label.setText(
            "Estado: predicción activa"
        )
    
        self.record_button.setEnabled(
            True
        )
        
        
    def stop_realtime_prediction(self):

        if self.realtime_thread is None:
    
            return
    
        if self.realtime_thread.isRunning():
    
            self.realtime_status_label.setText(
                "Estado: deteniendo..."
            )
    
            self.realtime_thread.stop()


    def realtime_stopped(self):
    
        self.realtime_display_timer.stop()
    
        #self.start_realtime_button.setEnabled(
        #    len(self.valid_models) > 0
        #)
        if self.pending_video_path is None:
        
            self.start_realtime_button.setEnabled(
                len(self.valid_models) > 0
            )
        
        else:
        
            self.start_realtime_button.setEnabled(
                False
            )
    
        self.stop_realtime_button.setEnabled(
            False
        )
    
        self.realtime_model_combo.setEnabled(
            True
        )
    
        self.camera_combo.setEnabled(
            True
        )
    
        self.refresh_camera_button.setEnabled(
            True
        )
    
        self.realtime_preview.clear()
    
        self.realtime_preview.setText(
            "Cámara detenida"
        )
    
        self.camera_fps_label.setText(
            "FPS cámara: —"
        )
    
        self.yolo_fps_label.setText(
            "FPS inferencia: —"
        )
    
        self.inference_time_label.setText(
            "Tiempo YOLO: —"
        )
    
        self.realtime_status_label.setText(
            "Estado: detenido"
        )
    
        self.statusBar().showMessage(
            "Predicción en tiempo real detenida"
        )
        
        # =====================================================
        # ¿Terminó una grabación y debemos procesarla?
        # =====================================================
    
        if self.pending_video_path is not None:
    
            self.start_video_processing()
            

    def realtime_error(
        self,
        message,
    ):
    
        QMessageBox.critical(
            self,
            "Error en tiempo real",
            message,
        )

    def start_recording(self):
    
        if self.realtime_thread is None:
    
            return
    
        if not self.realtime_thread.isRunning():
    
            QMessageBox.warning(
                self,
                "Grabación",
                "Primero inicie la cámara.",
            )
    
            return
    
        # =====================================================
        # Carpeta de salida
        # =====================================================
    
        output_root = (
            self.output_path_edit
            .text()
            .strip()
        )
    
        if not output_root:
    
            QMessageBox.warning(
                self,
                "Carpeta de salida",
                "Seleccione primero "
                "la carpeta de salida.",
            )
    
            return
    
        timestamp = (
            datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
        )
    
        session_folder = (
            Path(output_root)
            / "Video"
            / f"Grabacion_{timestamp}"
        )
    
        session_folder.mkdir(
            parents=True,
            exist_ok=True,
        )
    
        video_path = (
            session_folder
            / "original.mp4"
        )
    
        # =====================================================
        # Guardar información para procesamiento posterior
        # =====================================================
    
        self.pending_video_folder = (
            session_folder
        )
    
        self.pending_video_settings = (
            self.get_yolo_settings()
        )
    
        inference_fps = (
            self.inference_fps_combo.currentData()
        )
    
        if inference_fps is None:
    
            inference_fps = 1
    
        self.pending_video_inference_fps = (
            inference_fps
        )
    
        # =====================================================
        # Conectar señales
        # =====================================================
    
        try:
    
            self.realtime_thread.recording_started.disconnect(
                self.recording_started
            )
    
        except Exception:
    
            pass
    
        try:
    
            self.realtime_thread.recording_time_updated.disconnect(
                self.recording_time_updated
            )
    
        except Exception:
    
            pass
    
        try:
    
            self.realtime_thread.recording_finished.disconnect(
                self.recording_finished
            )
    
        except Exception:
    
            pass
    
        try:
    
            self.realtime_thread.recording_error.disconnect(
                self.recording_error
            )
    
        except Exception:
    
            pass
    
        self.realtime_thread.recording_started.connect(
            self.recording_started
        )
    
        self.realtime_thread.recording_time_updated.connect(
            self.recording_time_updated
        )
    
        self.realtime_thread.recording_finished.connect(
            self.recording_finished
        )
    
        self.realtime_thread.recording_error.connect(
            self.recording_error
        )
    
        # =====================================================
        # Grabar máximo 30 s
        # =====================================================
    
        self.realtime_thread.request_recording(
            output_path=video_path,
            max_seconds=30,
        )
    
        self.record_button.setEnabled(
            False
        )
    
        self.stop_record_button.setEnabled(
            True
        )
    
        # Evitamos detener toda la cámara accidentalmente
        self.stop_realtime_button.setEnabled(
            False
        )

    def recording_started(
        self,
        path,
    ):
    
        self.recording_active = True
    
        self.recording_time_label.setText(
            "Grabación: 00:00 / 00:30"
        )
    
        self.realtime_status_label.setText(
            "Estado: grabando video..."
        )

    def recording_time_updated(
        self,
        elapsed,
    ):
    
        elapsed = min(
            elapsed,
            30.0,
        )
    
        seconds = int(
            elapsed
        )
    
        self.recording_time_label.setText(
            f"Grabación: "
            f"00:{seconds:02d} / 00:30"
        )

    def stop_recording(self):
    
        if self.realtime_thread is None:
    
            return
    
        if not self.recording_active:
    
            return
    
        self.stop_record_button.setEnabled(
            False
        )
    
        self.realtime_status_label.setText(
            "Estado: finalizando grabación..."
        )
    
        self.realtime_thread.request_stop_recording()

    def recording_finished(
        self,
        video_path,
        duration,
    ):
    
        self.recording_active = False
    
        self.pending_video_path = (
            video_path
        )
    
        self.record_button.setEnabled(
            False
        )
    
        self.stop_record_button.setEnabled(
            False
        )
    
        self.realtime_status_label.setText(
            "Estado: grabación finalizada. "
            "Liberando modelo y cámara..."
        )
    
        # =====================================================
        # Detenemos tiempo real ANTES de procesar
        # todos los modelos
        # =====================================================
    
        if (
            self.realtime_thread
            is not None
            and self.realtime_thread.isRunning()
        ):
    
            self.realtime_thread.stop()


    def recording_error(
        self,
        message,
    ):
    
        self.recording_active = False
    
        self.record_button.setEnabled(
            True
        )
    
        self.stop_record_button.setEnabled(
            False
        )
    
        self.stop_realtime_button.setEnabled(
            True
        )
    
        QMessageBox.critical(
            self,
            "Error de grabación",
            message,
        )

    def start_video_processing(self):
    
        video_path = (
            self.pending_video_path
        )
    
        output_folder = (
            self.pending_video_folder
        )
    
        if (
            video_path is None
            or output_folder is None
        ):
    
            return
    
        self.video_models_progress.setValue(
            0
        )
    
        self.video_frame_progress.setValue(
            0
        )
    
        self.video_models_label.setText(
            "Preparando procesamiento..."
        )
    
        # =====================================================
        # Worker
        # =====================================================
    
        self.video_thread = (
            VideoPredictionThread(
                models=self.valid_models,
                video_path=video_path,
                output_folder=output_folder,
                settings=self.pending_video_settings,
                inference_fps=(
                    self.pending_video_inference_fps
                ),
                parent=self,
            )
        )
    
        self.video_thread.model_started.connect(
            self.video_model_started
        )
    
        self.video_thread.model_progress.connect(
            self.video_model_progress
        )
    
        self.video_thread.model_finished.connect(
            self.video_model_finished
        )
    
        self.video_thread.processing_finished.connect(
            self.video_processing_finished
        )
    
        self.video_thread.processing_error.connect(
            self.video_processing_error
        )
    
        self.video_thread.status_message.connect(
            self.statusBar().showMessage
        )
    
        self.video_thread.start()


    def video_model_started(
        self,
        model_name,
        current,
        total,
    ):
    
        self.video_models_label.setText(
            f"Modelo {current}/{total}: "
            f"{model_name}"
        )
    
        percent = int(
            (
                current - 1
            )
            / total
            * 100
        )
    
        self.video_models_progress.setValue(
            percent
        )
    
        self.video_frame_progress.setValue(
            0
        )

    def video_model_progress(
        self,
        current,
        total,
    ):
    
        if total <= 0:
    
            return
    
        percent = int(
            current
            / total
            * 100
        )
    
        self.video_frame_progress.setValue(
            percent
        )

    def video_model_finished(
        self,
        result,
    ):
    
        self.video_frame_progress.setValue(
            100
        )

    def video_processing_finished(self):
    
        self.video_models_progress.setValue(
            100
        )
    
        self.video_frame_progress.setValue(
            100
        )
    
        self.video_models_label.setText(
            "Procesamiento completado"
        )
    
        result_folder = (
            self.pending_video_folder
        )
    
        # Limpiar pendientes
        self.pending_video_path = None
    
        self.pending_video_folder = None
    
        self.pending_video_settings = None
    
        self.start_realtime_button.setEnabled(
            len(self.valid_models) > 0
        )
    
        self.realtime_status_label.setText(
            "Estado: procesamiento terminado"
        )
    
        QMessageBox.information(
            self,
            "Video procesado",
            (
                "El video fue procesado "
                "con todos los modelos válidos.\n\n"
                "Resultados:\n"
                f"{result_folder}"
            ),
        )

    def video_processing_error(
        self,
        message,
    ):
    
        self.pending_video_path = None
    
        self.start_realtime_button.setEnabled(
            len(self.valid_models) > 0
        )
    
        QMessageBox.critical(
            self,
            "Error procesando video",
            message,
        )











