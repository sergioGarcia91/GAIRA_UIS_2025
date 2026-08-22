# Requerimientos
El aplicativo fue desarrollado en `Python==3.10.20`.

```
ultralytics>=8.3.0
PySide6==6.6.3.1
opencv-python==4.10.0.84
numpy==1.26.4
```

# Ejecución

```bash
python -m PyInstaller --noconfirm --clean --onedir --name YOLO_MultiModel_Predictor --runtime-hook rthook_torch_first.py --collect-all ultralytics --collect-submodules ultralytics --collect-binaries torch --exclude-module PyQt5 --exclude-module PyQt6 --exclude-module PySide2 main.py
```

