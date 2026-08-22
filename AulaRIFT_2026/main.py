# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 19:35:32 2026

@author: thesa
"""

import sys

# ============================================================
# IMPORTANTE EN WINDOWS
# PyTorch debe cargarse antes de PySide6 para evitar
# conflictos con las DLL de torch, especialmente c10.dll
# ============================================================

import torch

from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow


def main():

    # --------------------------------------------------------
    # Spyder ya puede tener una QApplication creada.
    # Desde CMD normalmente no existe.
    # --------------------------------------------------------

    app = QApplication.instance()

    app_created = False

    if app is None:

        app = QApplication(
            sys.argv
        )

        app_created = True

    app.setApplicationName(
        "YOLO Multi-Model Predictor"
    )

    # --------------------------------------------------------
    # Ventana principal
    # --------------------------------------------------------

    window = MainWindow()

    window.show()

    # --------------------------------------------------------
    # Desde CMD debemos iniciar el event loop.
    # En Spyder ya existe.
    # --------------------------------------------------------

    if app_created:

        sys.exit(
            app.exec()
        )

    return window


if __name__ == "__main__":

    window = main()