# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 19:36:42 2026

@author: thesa
"""

from dataclasses import dataclass


@dataclass
class YoloSettings:
    conf: float = 0.25
    iou: float = 0.70
    imgsz: int = 640
    max_det: int = 300
    device: str = "auto"
    half: bool = False