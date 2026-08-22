# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 07:47:44 2026

@author: thesa
"""

import os
import sys


# ============================================================
# PyTorch debe inicializarse antes que PySide6 en Windows
# ============================================================

if sys.platform == "win32":

    # En una aplicación congelada por PyInstaller,
    # sys._MEIPASS apunta a la carpeta interna del bundle.
    if getattr(sys, "frozen", False):

        torch_lib = os.path.join(
            sys._MEIPASS,
            "torch",
            "lib",
        )

        if os.path.isdir(torch_lib):

            # Priorizar las DLL de PyTorch
            os.environ["PATH"] = (
                torch_lib
                + os.pathsep
                + os.environ.get("PATH", "")
            )

            # Mantener el directorio registrado para búsqueda de DLL
            if hasattr(os, "add_dll_directory"):

                _torch_dll_handle = os.add_dll_directory(
                    torch_lib
                )


# IMPORTANTE:
# PyTorch se importa aquí, antes de que PyInstaller
# ejecute el runtime hook de PySide6.
import torch