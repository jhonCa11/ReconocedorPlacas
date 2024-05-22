# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from alpr.detector import PlateDetector
import cv2
from timeit import default_timer as timer
from argparse import ArgumentParser
#Codigo para ejecutar: python detector_demo.py --fuente-video ./assets/test_patente2.mp4 --mostrar-resultados --input-size 608


def main_demo(args):
    # Configuración de los parámetros
    input_size = args.input_size
    video_path = args.video_source
    weights_path = f'./alpr/models/detection/tf-yolo_tiny_v4-{input_size}x{input_size}-custom-anchors/'
    iou = 0.45
    score = 0.25
    
    # Inicialización del detector de placas
    detector_patente = PlateDetector(
        weights_path, input_size=input_size, iou=iou, score=score)
    print("Video from: ", video_path)
    vid = cv2.VideoCapture(video_path)

    frame_id = 0
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            break
        
        # Preprocesamiento del frame
        input_img = detector_patente.preprocess(frame)
        
        # Inferencia
        yolo_out = detector_patente.predict(input_img)
        
        # Obtención de los bounding boxes después de NMS
        bboxes = detector_patente.procesar_salida_yolo(yolo_out)
        
        # Mostrar predicciones
        start = timer()
        frame_w_preds = detector_patente.draw_bboxes(frame, bboxes)
        end = timer()
        
        # Tiempo de inferencia
        exec_time = end - start
        fps = 1. / exec_time
        
        # Mostrar benchmark si está habilitado
        if args.mostrar_benchmark:
            print(f'Inferencia\tms: {exec_time:.5f}\t', end='')
            print(f'FPS: {fps:.0f}')
        
        # Mostrar resultados si está habilitado
        if args.mostrar_resultados:
            result = cv2.cvtColor(frame_w_preds, cv2.COLOR_RGB2BGR)
            # Mostrar resultados
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_id += 1


if __name__ == '__main__':
    try:
        parser = ArgumentParser()
        parser.add_argument("-f", "--fuente-video", dest="video_source",
                            required=True, type=str, help="Video de entrada, para video: 0,\
                                camara ip: rtsp://user:pass@IP:Puerto, video en disco: C:/.../vid.mp4")
        parser.add_argument("-i", "--input-size", dest="input_size",
                            default=512, type=int, help="Modelo a usar, opciones: 384, 512, 608")
        parser.add_argument("-m", "--mostrar-resultados", dest="mostrar_resultados",
                            action='store_true', help="Mostrar los frames con las patentes dibujadas")
        parser.add_argument("-b", "--benchmark", dest="mostrar_benchmark",
                            action='store_true', help="Mostrar tiempo de inferencia (ms y FPS)")
        args = parser.parse_args()
        main_demo(args)
    except Exception as e:
        print(e)
