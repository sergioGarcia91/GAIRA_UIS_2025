# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 19:37:36 2026

@author: thesa
"""

from dataclasses import dataclass

import platform

import cv2


@dataclass
class CameraInfo:

    index: int

    width: int
    height: int

    fps: float

    backend: str


    @property
    def display_name(self):

        resolution = (
            f"{self.width}x{self.height}"
            if self.width and self.height
            else "Resolución desconocida"
        )

        fps_text = (
            f"{self.fps:.1f} FPS"
            if self.fps > 0
            else "FPS desconocidos"
        )

        return (
            f"Cámara {self.index} — "
            f"{resolution} — "
            f"{fps_text}"
        )


def backend_candidates():
    """
    Selecciona diferentes métodos para abrir
    cámaras dependiendo del sistema operativo.
    """

    system = platform.system().lower()

    # -----------------------------
    # Windows
    # -----------------------------

    if system == "windows":

        return [

            (
                cv2.CAP_DSHOW,
                "DirectShow",
            ),

            (
                cv2.CAP_MSMF,
                "Media Foundation",
            ),

            (
                cv2.CAP_ANY,
                "Automático",
            ),
        ]

    # -----------------------------
    # Linux
    # -----------------------------

    if system == "linux":

        return [

            (
                cv2.CAP_V4L2,
                "V4L2",
            ),

            (
                cv2.CAP_ANY,
                "Automático",
            ),
        ]

    # -----------------------------
    # Mac
    # -----------------------------

    if system == "darwin":

        return [

            (
                cv2.CAP_AVFOUNDATION,
                "AVFoundation",
            ),

            (
                cv2.CAP_ANY,
                "Automático",
            ),
        ]

    return [
        (
            cv2.CAP_ANY,
            "Automático",
        )
    ]


def open_camera(index):
    """
    Intenta abrir una cámara utilizando
    diferentes backends.
    """

    for backend, backend_name in backend_candidates():

        cap = cv2.VideoCapture(
            index,
            backend,
        )

        if cap.isOpened():

            success, frame = cap.read()

            if success:

                return cap, backend_name

        cap.release()

    return None, ""


def detect_cameras(max_index=9):
    """
    Busca cámaras desde el puerto 0
    hasta max_index.
    """

    cameras = []

    for index in range(max_index + 1):

        cap, backend_name = open_camera(index)

        if cap is None:
            continue

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
            or 0
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
            or 0
        )

        fps = float(
            cap.get(
                cv2.CAP_PROP_FPS
            )
            or 0
        )

        camera = CameraInfo(
            index=index,
            width=width,
            height=height,
            fps=fps,
            backend=backend_name,
        )

        cameras.append(camera)

        cap.release()

    return cameras