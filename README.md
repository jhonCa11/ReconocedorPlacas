**ConvALPR**, o **Convolutional Automatic License Plate Recognition**, es una herramienta innovadora para el reconocimiento automático de placas vehiculares que aprovecha el poder de las **Redes Neuronales Convolucionales (CNNs)**. A diferencia de los métodos tradicionales, ConvALPR es capaz de reconocer placas incluso en condiciones adversas como obstrucciones, variaciones en el brillo, letras borrosas, entre otros desafíos.

El sistema ConvALPR consta de dos procesos principales: la localización de placas utilizando un detector de objetos y el **reconocimiento óptico de caracteres (OCR)**. Ambos procesos se basan exclusivamente en redes neuronales convolucionales, también conocidas como **ConvNets** o **CNNs**. Este enfoque permite una mayor precisión y robustez en el reconocimiento de placas, ya que las CNNs son capaces de aprender y adaptarse a una amplia variedad de condiciones y patrones presentes en las imágenes de las placas vehiculares

## Modo de Uso

### Instalación de Dependencias

Con Python versión: **3.x**:

```
pip install .
```

Para aprovechar la potencia de la **placa de video/GPU** y acelerar la inferencia, es necesario instalar los siguientes **[requisitos](https://www.tensorflow.org/install/gpu#software_requirements)**.

### Visualización del Localizador

Para probar únicamente el **localizador/detector** de placas (**sin OCR, solo los cuadros delimitadores**) y visualizar las predicciones, se utiliza el siguiente comando:

```
python detector_demo.py --fuente-video /ruta/alu¿/video.mp4 --mostrar-resultados --input-size 608
```

*Se puede experimentar con distintos modelos {608, 512, 384} para encontrar el que mejor se adapte.*

## Reconocimiento Automático

### Configuración

La **configuración** del **ALPR** se encuentra detallada en el archivo **config.yaml**. Este archivo alberga los parámetros ajustables tanto del reconocedor como del localizador, presentando una descripción minuciosa de cada opción disponible. Es importante destacar que el modelo de OCR opera de manera **independiente** al detector de objetos, lo que garantiza su compatibilidad con cualquier configuración seleccionada.


### Ejemplo de Visualización del ALPR

```
python reconocedor_automatico.py --cfg config.yaml --demo
```

### Guardar en Base de Datos sin Visualizar

```
python reconocedor_automatico.py --cfg config.yaml
```