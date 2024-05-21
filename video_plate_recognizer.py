import cv2
import numpy as np

class PlateOCR:
    """
    Módulo encargado del reconocimiento
    de caracteres de las placas (ya recortadas)
    """

    def __init__(self, confidence_threshold: float = 0.5, none_low_threshold: float = 0.35) -> None:
        self.confidence_threshold = confidence_threshold
        self.none_low_threshold = none_low_threshold
        # Aquí inicializarías tu modelo de reconocimiento de placas

    def predict(self, iter_coords, frame: np.ndarray) -> list:
        """
        Reconoce a partir de un frame todas
        las placas en formato de texto

        Parámetros:
            iter_coords:    objeto generador que genera las placas
            frame:  sub-frame conteniendo la placa candidata
        Retorna:
            Lista de placas (en formato de texto)
        """
        plates = []
        for yolo_prediction in iter_coords:
            x1, y1, x2, y2, _ = yolo_prediction
            plate, probs = self.predict_ocr(x1, y1, x2, y2, frame)
            avg = np.mean(probs)
            if avg > self.confidence_threshold and self.none_low(probs[0], thresh=self.none_low_threshold):
                plate = ("".join(plate)).replace("_", "")
                plates.append(plate)
        return plates

    def none_low(self, probs, thresh=0.5):
        """
        Devuelve False si hay algún caracter
        con probabilidad por debajo de thresh
        """
        if isinstance(probs, np.ndarray):
            probs = probs.flatten()  
        return all(prob >= thresh for prob in probs)

    def predict_ocr(self, x1: int, y1: int, x2: int, y2: int, frame: np.ndarray):
        """
        Hace OCR en un sub-frame del frame

        Parámetros:
            x1: Valor de x de la esquina superior izquierda del rectángulo
            y1:    "     y           "             "                  "
            x2: Valor de x de la esquina inferior derecha del rectángulo
            y2:    "     y           "             "                  "
            frame: array conteniendo la imagen original
        """
        # Aquí iría tu lógica de reconocimiento de placas
        # En este ejemplo, simplemente devolvemos un texto ficticio y una lista de probabilidades
        return ["ABC123"], [[0.9, 0.8, 0.7, 0.6, 0.5]]

class VideoPlateRecognizer:
    """
    Clase para procesar videos en busca de placas.
    """

    def __init__(self) -> None:
        self.plate_ocr = PlateOCR()

    def detect_plates(self, video_path: str) -> list:
        """
        Detecta placas en un video.

        Args:
            video_path (str): Ruta del archivo de video.

        Returns:
            List: Lista de coordenadas de las placas detectadas en el video.
        """
        plates = []

        # Captura de video
        cap = cv2.VideoCapture(video_path)

        # Verificar si la captura de video se abrió correctamente
        if not cap.isOpened():
            print("Error al abrir el video")
            return plates

        # Inicio de la captura de video
        while cap.isOpened():
            ret, frame = cap.read()

            # Verificar si se pudo leer el frame correctamente
            if not ret:
                break

            # Detección de placas (aquí deberías implementar tu lógica de detección)
            plate_coords = [(10, 10, 100, 100)]  # Lista de coordenadas de ejemplo
            plates.append(plate_coords)

        # Liberar los recursos
        cap.release()

        return plates

    def process_video(self, video_path: str, output_path: str) -> None:
        """
        Procesa un video en busca de placas y guarda el resultado en otro video.

        Args:
            video_path (str): Ruta del archivo de video de entrada.
            output_path (str): Ruta del archivo de video de salida.
        """
        plates = self.detect_plates(video_path)

        # Comprueba si se detectaron placas en el video
        if plates:
            # Si se detectaron placas, aquí puedes agregar la lógica para dibujar las
            # placas en el video y guardar el resultado
            print("Placas detectadas:", plates)
        else:
            print("No se detectaron placas en el video.")

if __name__ == "__main__":
    recognizer = VideoPlateRecognizer()
    recognizer.process_video("assets/test_patente1.mp4", "Results/video_procesado.mp4")
