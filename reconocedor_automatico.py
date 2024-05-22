import os
# Mostrar solo errores de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Deshabilitar GPU (correr en CPU)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from alpr.alpr import ALPR
from argparse import ArgumentParser
import yaml
import logging
from timeit import default_timer as timer
import cv2
from datetime import datetime

# Configurar el registro de logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main_demo(cfg, demo=True, benchmark=True, save_vid=False):
    # Inicializar el objeto ALPR
    alpr = ALPR(cfg['modelo'])
    video_path = cfg['video']['fuente']
    
    # Obtener la fecha y hora actual para usarla en el nombre del video
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Abrir el video
    cap = cv2.VideoCapture(video_path)
    is_img = cv2.haveImageReader(video_path)
    cv2_wait = 0 if is_img else 1
    
    logger.info(f'Se va analizar la fuente: {video_path}')
    frame_id = 0
    
    # Guardar el video si está habilitado
    if save_vid:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        size = (width, height)
        # Agregar la fecha y hora actual al nombre del video
        video_name = f'Results/resultado_{current_time}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_name, fourcc, 20.0, size)
    
    # Frecuencia de inferencia
    intervalo_reconocimiento = cfg['video']['frecuencia_inferencia']
    if not is_img:
        logger.info(f'El intervalo del reconocimiento para el video es de: {intervalo_reconocimiento}')
    
    while True:
        return_value, frame = cap.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            # Para camara IP 
            #Esto es por si el stream deja de transmitir algún frame o se tarda más de lo normal. 
            #En este caso simplemente volvemos a intentar leer el frame.
            # vid = cv2.VideoCapture(video_path)
            # continue
            break
        
        # Demostrar o predecir
        if demo:
            # Mostrar predicciones
            frame_w_pred, total_time = alpr.mostrar_predicts(frame)
            frame_w_pred = cv2.cvtColor(frame_w_pred, cv2.COLOR_RGB2BGR)
            frame_w_pred_r = cv2.resize(frame_w_pred, dsize=(1400, 1000))
            
            if benchmark:
                display_bench = f'ms: {total_time:.4f} FPS: {1 / total_time:.0f}'
                fontScale = 1.5
                cv2.putText(frame_w_pred_r, display_bench, (5, 45), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, (10, 140, 10), 4)
            
            if save_vid:
                out.write(frame_w_pred)
            
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("result", frame_w_pred_r)
            
            if cv2.waitKey(cv2_wait) & 0xFF == ord('q'):
                break
        else:
            # Predecir solo si el frame_id es múltiplo del intervalo_reconocimiento
            if frame_id % intervalo_reconocimiento == 0:
                start = timer()
                alpr.predict(frame)
                total_time = timer() - start
                
                if benchmark:
                    display_bench = f'ms: {total_time:.4f} FPS: {1 / total_time:.0f}'
                    print(display_bench, flush=True)
        
        frame_id += 1
    
    cap.release()
    
    if save_vid:
        out.release()
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        parser = ArgumentParser()
        parser.add_argument("--cfg", dest="cfg_file", help="Path del archivo de config, default: ./config.yaml", default='config.yaml')
        parser.add_argument("--demo", dest="demo", action='store_true', help="En vez de guardar las patentes, mostrar las predicciones")
        parser.add_argument("--guardar_video", dest="save_video", action='store_true', help="Guardar video en ./Results/resultado.mp4")
        parser.add_argument("--benchmark", dest="bench", action='store_true', help="Medir la inferencia (incluye todo el pre/post processing")
        args = parser.parse_args()
        
        with open(args.cfg_file, 'r') as stream:
            try:
                cfg = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logger.exception(exc)
        
        main_demo(cfg, args.demo, args.bench, args.save_video)
    except Exception as e:
        logger.exception(e)
