from alpr.alpr import ALPR

import cv2
import yaml


im = cv2.imread('assets/pruebaa1.jpg')
with open('config.yaml', 'r') as stream:
    cfg = yaml.safe_load(stream)
alpr = ALPR(cfg['modelo'])
predicciones = alpr.predict(im)
print(predicciones)

"""
im = cv2.imread('assets/pruebaa1.jpg')
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


