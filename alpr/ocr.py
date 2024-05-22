"""
OCR module.
"""

import cv2
import numpy as np
from fast_plate_ocr import ONNXPlateRecognizer

class PlateOCR:
    """
    Módulo encargado del reconocimiento
    de caracteres de las patentes (ya recortadas)
    """

    def __init__(self, confianza_avg: float = 0.5, none_low_thresh: float = 0.35) -> None:
        # Inicialización del modelo OCR
        self.confianza_avg = confianza_avg
        self.none_low_thresh = none_low_thresh
        self.ocr_model = ONNXPlateRecognizer("argentinian-plates-cnn-model")

    def predict(self, iter_coords, frame: np.ndarray) -> list:
        """
        Reconoce todas las patentes en formato de texto a partir de un frame.

        Parámetros:
            iter_coords: generator object que produce las coordenadas de las patentes
            frame: sub-frame conteniendo la patente candidata
        Devuelve:
            Lista de patentes (en formato de texto)
        """
        patentes = []
        for yolo_prediction in iter_coords:
            x1, y1, x2, y2, _ = yolo_prediction
            plate, probs = self.predict_ocr(x1, y1, x2, y2, frame)
            avg = np.mean(probs)
            # Ignora las patentes con baja confianza
            if avg > self.confianza_avg and self.none_low(probs[0], thresh=self.none_low_thresh):
                plate = ("".join(plate)).replace("_", "")
                patentes.append(plate)
        return patentes

    def none_low(self, probs, thresh=0.5):
        """
        Devuelve False si hay algún carácter con probabilidad por debajo de thresh.
        """
        if isinstance(probs, np.ndarray):
            probs = probs.flatten()  # Aplanar el array a una dimensión
        return all(prob >= thresh for prob in probs)

    def predict_ocr(self, x1: int, y1: int, x2: int, y2: int, frame: np.ndarray):
        """
        Realiza OCR en un sub-frame del frame.

        Parámetros:
            x1: Valor de x de la esquina superior izquierda del rectángulo
            y1: Valor de y de la esquina superior izquierda del rectángulo
            x2: Valor de x de la esquina inferior derecha del rectángulo
            y2: Valor de y de la esquina inferior derecha del rectángulo
            frame: array conteniendo la imagen original
        Devuelve:
            Texto de la placa reconocida y las probabilidades asociadas
        """
        cropped_plate = frame[y1:y2, x1:x2]
        cropped_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
        plate, probs = self.ocr_model.run(cropped_plate, return_confidence=True)
        # Imprime el texto de la placa y las probabilidades
        print("Texto de la placa:", plate)
        print("Probabilidades:", probs)
        return plate, probs 
