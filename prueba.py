from alpr.alpr import ALPR

import cv2
import yaml
from video_plate_recognizer import VideoPlateRecognizer


recognizer = VideoPlateRecognizer()
recognizer.process_video("assets/test_patente1.mp4", "Results/video_procesado.mp4")

"""
im = cv2.imread('assets/prueba1.jpg')
with open('config.yaml', 'r') as stream:
    cfg = yaml.safe_load(stream)
alpr = ALPR(cfg, model_id="colombian-plates/3", api_key="QTIoT4teH3JdSCbzDqSN")
predicciones = alpr.predict(im)
print(predicciones)
"""
"""
from alpr.alpr import ALPR
import cv2

im = cv2.imread('assets/prueba1.jpg')
alpr = ALPR(
    {
        'resolucion_detector': 512,
        'confianza_detector': 0.25,
        'numero_modelo_ocr': 2,
        'confianza_avg_ocr': .4,
        'confianza_low_ocr': .35
    }
)
predicciones = alpr.predict(im)
print(predicciones)
"""
