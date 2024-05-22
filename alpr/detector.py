"""
Plate Detector.
"""

import cv2
import numpy as np
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


class PlateDetector:
    """
    Módulo encargado del detector de patentes
    """

    def __init__(
        self, weights_path: str, input_size: int = 608, iou: float = 0.45, score: float = 0.25
    ):
        # Inicialización de los parámetros del detector de placas
        self.input_size = input_size
        self.iou = iou
        self.score = score
        # Cargar el modelo YOLO
        self.saved_model_loaded = tf.saved_model.load(weights_path)
        self.yolo_infer = self.saved_model_loaded.signatures["serving_default"]

    def procesar_salida_yolo(self, output):
        """
        Aplica Non Max Suppression (NMS) a la salida de YOLO
        para eliminar detecciones duplicadas.

        Parámetros:
            output: tensor con la salida de YOLO
        Devuelve:
            Lista con los bounding boxes de las patentes detectadas después de NMS
        """
        # Aplicar NMS a la salida de YOLO
        for value in output.values():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=self.iou,
            score_threshold=self.score,
        )
        return [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

    def preprocess(self, frame):
        """
        Preprocesa la imagen para la entrada al modelo YOLO.

        Parámetros:
            frame: numpy array conteniendo la imagen original
        Devuelve:
            Tensor preprocesado
        """
        # Normalizar pixeles y ajustar dimensiones para YOLO
        image_data = cv2.resize(frame, (self.input_size, self.input_size))
        image_data = image_data / 255.0
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        return tf.constant(image_data)

    def predict(self, input_img: tf.Tensor):
        """
        Realiza la inferencia con el modelo YOLO.

        Parámetros:
            input_img: tensor con dimensiones (1, self.input_size, self.input_size, 3)
        Devuelve:
            Salida de YOLO
        """
        return self.yolo_infer(input_img)

    def draw_bboxes(self, frame: np.ndarray, bboxes: list):
        """
        Dibuja los bounding boxes de las patentes en el frame.

        Parámetros:
            frame: numpy array conteniendo el frame original
            bboxes: predicciones/output de después del NMS
        Devuelve:
            Frame con los bounding boxes dibujados
        """
        # Dibujar los bounding boxes en el frame
        for x1, y1, x2, y2, score in self.yield_coords(frame, bboxes):
            font_scale = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 255, 12), 2)
            cv2.putText(
                frame,
                f"{score:.2f}%",
                (x1, y1 - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (20, 10, 220),
                5,
            )
        return frame

    def yield_coords(self, frame: np.ndarray, bboxes: list):
        """
        Genera las coordenadas de los bounding boxes.

        Parámetros:
            frame: numpy array conteniendo el frame original
            bboxes: predicciones/output de después del NMS
        Devuelve:
            Coordenadas de los bounding boxes y la probabilidad de objectness
        """
        # Obtener las coordenadas de los bounding boxes
        out_boxes, out_scores, _, num_boxes = bboxes
        image_h, image_w, _ = frame.shape
        for i in range(num_boxes[0]):
            coor = out_boxes[0][i]
            x1 = int(coor[1] * image_w)
            y1 = int(coor[0] * image_h)
            x2 = int(coor[3] * image_w)
            y2 = int(coor[2] * image_h)
            yield x1, y1, x2, y2, out_scores[0][i]
