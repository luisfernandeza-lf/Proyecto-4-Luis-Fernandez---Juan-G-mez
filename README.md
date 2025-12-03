# Proyecto 4, Luis Eduardo Fernández Anaya y Juan José Gómez Correa
## Primera parte del código
### Esta primera sección se encarga de importar las bibliotecas necesarias (cv2 para el procesamiento de imágenes/video, numpy para operaciones numéricas, y matplotlib.pyplot y cv2_imshow para visualización en entornos como Google Colab). Luego, solicita al usuario que cargue un archivo de video a través de la función files.upload(), almacena la ruta y utiliza cv2.VideoCapture() para abrir el video. Finalmente, verifica si la apertura fue exitosa e imprime un mensaje de confirmación o error.
```
import cv2 # Importa la biblioteca OpenCV para procesamiento de visión artificial
import numpy as np # Importa NumPy para manejo eficiente de arrays y operaciones numéricas
import matplotlib.pyplot as plt # Importa Matplotlib para posibles gráficos, aunque su uso es mínimo aquí
from google.colab.patches import cv2_imshow # Función específica para mostrar imágenes/frames en Google Colab
```
### Carga de Archivo de Video
```
from google.colab import files
uploaded = files.upload() # Abre una ventana para que el usuario suba un archivo (específico de Colab)
video_path = next(iter(uploaded)) # Obtiene el nombre del archivo subido (la clave del diccionario 'uploaded')
```
### Inicializa el objeto VideoCapture para leer el video
```
cap = cv2.VideoCapture(video_path)
```
### Verifica si el video se abrió correctamente
```
if not cap.isOpened():
    print("Error al abrir el video")
else:
    print("Video cargado correctamente")
```
### Esta sección define dos funciones cruciales. preprocess(frame) prepara cada frame del video: lo redimensiona a un tamaño estándar (640x360), lo convierte a escala de grises y aplica un filtro Gaussiano para reducir el ruido. detect_shot(...) es el corazón de la detección: calcula la diferencia absoluta entre el frame actual (su versión desenfocada) y el frame anterior (también desenfocado), umbraliza esta diferencia para resaltar el movimiento, y luego suma los píxeles blancos (movimiento) en una pequeña región central de la imagen. Si el movimiento en el centro supera un umbral (8000), se declara que se ha detectado un "tiro" o golpe.
```
def preprocess(frame):
```
### Convierte a gris, reduce ruido y estandariza tamaño
### Redimensiona el frame a 640x360 para estandarizar el procesamiento
```
    frame_resized = cv2.resize(frame, (640, 360))
```
### Convierte el frame redimensionado de BGR (color) a escala de grises
```
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
```
### Aplica un filtro Gaussiano (5x5) para suavizar la imagen y reducir el ruido
```
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
```
### Devuelve el frame original redimensionado, la versión en gris, y la versión desenfocada
```
    return frame_resized, gray, blur
```
```
def detect_shot(previous_frame, current_frame, threshold=25):
```
### Detecta tiros basado en diferencia entre frames
### Calcula la diferencia absoluta de píxeles entre el frame anterior y el actual
```
    diff = cv2.absdiff(previous_frame, current_frame)
```
### Aplica un umbral (threshold=25) para convertir las diferencias significativas en blanco (255)
### Esto crea un mapa de movimiento binario
```
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
```
### Definición de la Región Central de Interés (ROI)
```
    h, w = thresh.shape
```
### Define las coordenadas X para una región central de 60 píxeles de ancho
```
    cx1, cx2 = w//2 - 30, w//2 + 30
```
### Define las coordenadas Y para una región central de 60 píxeles de alto
```
    cy1, cy2 = h//2 - 30, h//2 + 30
```
### Extrae la región central del mapa de movimiento umbralizado
```
    center_region = thresh[cy1:cy2, cx1:cx2]
```
### Suma los valores de píxeles en la región central (donde 255 es movimiento)
### Si la suma supera 8000, se considera un "tiro" detectado
```
    shot_detected = np.sum(center_region) > 8000
```
### Devuelve el estado de detección y el mapa de diferencia umbralizado
```
    return shot_detected, thresh
```
### Esta última parte establece el bucle principal que procesa el video frame por frame. Primero, lee el primer frame para inicializar la variable prev_blur. Dentro del bucle while True, lee un frame (frame), lo preprocesa, y llama a detect_shot para comparar el frame actual (blur) con el anterior (prev_blur). Si se detecta un tiro, dibuja un círculo rojo en el centro del original frame como marcador visual. Muestra el frame resultante cada 30 frames (aproximadamente una vez por segundo, dependiendo del framerate del video) usando cv2_imshow. Finalmente, actualiza prev_blur con el frame actual para la siguiente iteración, e imprime la cantidad total de frames procesados al terminar.

```
frames_processed = 0 # Contador para llevar la cuenta de los frames procesados
```
### Inicialización del primer frame para la comparación
```
ret, prev_frame = cap.read()
if not ret:
    print("No se pudo leer el primer frame")
else:
    # Preprocesa el primer frame y guarda solo la versión desenfocada (blur)
    _, prev_gray, prev_blur = preprocess(prev_frame)
```
### Bucle de Procesamiento de Video
```
while True:
    ret, frame = cap.read() # Lee el siguiente frame del video
    if not ret:
        break # Sale del bucle si no se puede leer un frame (fin del video)
```
### Preprocesa el frame actual
```
    original, gray, blur = preprocess(frame)
```
### Llama a la función de detección comparando el frame anterior y el actual (versiones 'blur')
```
    shot, diff_map = detect_shot(prev_blur, blur)

    if shot:
        # Si se detecta un tiro, dibuja un círculo rojo en el centro del frame original
        cv2.circle(original, (320, 180), 20, (0, 0, 255), 3)
```
### Muestra el frame resultante cada 30 frames (para no sobrecargar la visualización)
```
    if frames_processed % 30 == 0:
        cv2_imshow(original)
```
### Actualiza 'prev_blur' al frame actual para la próxima iteración
```
    prev_blur = blur
    frames_processed += 1 # Incrementa el contador de frames
```
### Libera el objeto VideoCapture al terminar (buena práctica)
```
cap.release()

print(f"Total de frames procesados: {frames_processed}")
```
## Segunda parte del código
### Esta primera sección se dedica a la preparación del entorno y la carga de los recursos necesarios. Primero, instala la librería ultralytics (que incluye YOLOv8). Luego, importa las bibliotecas estándar de procesamiento de visión (OpenCV) y utilidades (NumPy, Colab/Matplotlib). El usuario carga un archivo de video. La parte crucial es la carga del modelo de detección de objetos: model = YOLO("yolov8n.pt") utiliza la versión "nano" de YOLOv8, pre-entrenada para detectar clases comunes, incluida la clase "person" (persona), que aquí se interpreta como el "enemigo".
```
!pip install ultralytics # Instala la librería que contiene el modelo YOLO
from ultralytics import YOLO # Importa la clase YOLO para la detección de objetos
import cv2 # Importa OpenCV
import numpy as np # Importa NumPy
from google.colab.patches import cv2_imshow # Función para mostrar imágenes en Colab
from google.colab import files # Función para subir archivos en Colab
import matplotlib.pyplot as plt # Importa Matplotlib para la generación de gráficos
```
### SUBIR VIDEO
```
uploaded = files.upload() # Solicita al usuario subir el archivo de video
video_path = next(iter(uploaded)) # Obtiene el nombre del archivo

cap = cv2.VideoCapture(video_path) # Inicializa el objeto VideoCapture

if not cap.isOpened():
    print("Error al abrir video")
else:
    print("Video cargado correctamente")
```
### CARGAR MODELO YOLO
### Carga el modelo pre-entrenado YOLOv8n (nano), que es eficiente y bueno para detectar personas
```
model = YOLO("yolov8n.pt")
```
### FUNCIÓN DE MIRA
### Función simple para calcular las coordenadas del centro de la pantalla (la "mira")
```
def get_crosshair(frame):
    h, w, _ = frame.shape
    return w//2, h//2
```
### CONTADORES 
### Inicialización de contadores para estadísticas
```
total_frames = 0 # Contador de todos los frames leídos
processed_frames = 0 # Contador de frames realmente analizados (solo 1 de cada 10)

aim_on_target = 0 # Cuenta las veces que la mira está sobre un enemigo
aim_head = 0 # Cuenta aciertos en la cabeza
aim_torso = 0 # Cuenta aciertos en el torso
aim_legs = 0 # Cuenta aciertos en las piernas
```
### Esta sección contiene el bucle principal que itera sobre los frames del video. Para optimizar el rendimiento, solo se analiza 1 de cada 10 frames. Dentro del bucle:

### 1. Redimensiona el frame y calcula la posición central (mira).
### 2. Ejecuta el modelo YOLO (model(frame, conf=0.50, ...)), detectando objetos con una confianza mínima del 50%.
### 3. Itera sobre los resultados: si el objeto detectado es una "person", calcula la caja delimitadora (x1, y1, x2, y2).
### 4. Evalúa el acierto (Aiming): Verifica si las coordenadas de la mira (cx, cy) caen dentro de la caja de la persona.
### 5. Evalúa la zona de impacto: Si hay acierto, segmenta la caja de la persona en cabeza (25% superior), torso (25% al 60% de altura) y piernas (resto), incrementando el contador correspondiente (aim_head, aim_torso, etc.) y dibujando el resultado en el frame.
### 6. Solo muestra el frame si se detectó un enemigo, si se estaba apuntando o si se registró un impacto.
### PROCESAMIENTO
```
while True:
    ret, frame = cap.read() # Lee el siguiente frame
    if not ret:
        break # Sale si termina el video

    total_frames += 1 # Incrementa el contador total
```
### Analiza solo 1 de cada 10 frames para optimizar la velocidad
```
    if total_frames % 10 != 0:
        continue

    processed_frames += 1 # Incrementa el contador de frames analizados

    frame = cv2.resize(frame, (640, 360)) # Redimensiona el frame para el análisis
    h, w, _ = frame.shape
    cx, cy = get_crosshair(frame) # Calcula la posición de la mira

    cv2.circle(frame, (cx, cy), 4, (0,255,255), -1) # Dibuja la mira (círculo amarillo)
```
### Ejecuta el modelo YOLO en el frame con 50% de confianza mínima
```
    results = model(frame, conf=0.50, verbose=False)

    aimed = False # Bandera: ¿La mira está sobre algún enemigo?
    enemy_detected = False # Bandera: ¿Hay algún enemigo en el frame?
    hit_detected = False # Bandera: ¿Se registró un impacto en este frame?
```
### DETECTAR ENEMIGOS
```
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0]) # Obtiene el ID de la clase

            if model.names[cls] != "person":
                continue # Solo procesa si la clase detectada es "person"

            enemy_detected = True

            x1, y1, x2, y2 = map(int, box.xyxy[0]) # Coordenadas de la caja delimitadora
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2) # Dibuja el recuadro del enemigo
```
### Calcula los límites para segmentar la zona de impacto
```
            height = y2 - y1
```
### Cabeza (25% superior de la caja)
```
            head_limit  = y1 + int(height * 0.25)
```
### Torso (hasta el 60% de la altura de la caja)
```
            torso_limit = y1 + int(height * 0.60)
```
### ¿La mira está dentro del enemigo?
```
            if x1 < cx < x2 and y1 < cy < y2:
                aimed = True
```
### Determina la zona de impacto
```
                if cy <= head_limit:
                    aim_head += 1
                    zone = "HEADSHOT"
                    color = (0,0,255) # Rojo
                elif cy <= torso_limit:
                    aim_torso += 1
                    zone = "BODYSHOT"
                    color = (0,255,0) # Verde
                else:
                    aim_legs += 1
                    zone = "LEGSHOT"
                    color = (255,0,0) # Azul

                hit_detected = True
```
### Muestra el texto de la zona de impacto en el frame
```
                cv2.putText(frame, zone, (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

                break # Deja de analizar más cajas si ya se encontró un impacto
```
### Actualizar contador AIM si se apuntó a un enemigo en el frame
```
    if aimed:
        aim_on_target += 1
    else:
        # Si no se apunta a ningún enemigo, muestra el mensaje
        cv2.putText(frame, "NOT AIMING", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
```
### MOSTRAR FRAME solo si hay actividad relevante
```
    if enemy_detected or aimed or hit_detected:
        cv2_imshow(frame)

cap.release() # Libera el objeto de captura de video
```
### Una vez que el bucle ha terminado de procesar el video, esta sección calcula y muestra las estadísticas de precisión y la distribución de los aciertos.
### 1. Calcula el porcentaje total de frames en los que el jugador estuvo apuntando a un enemigo (p_total_aim).
### 2. Si se registraron aciertos válidos (valid_hits), calcula la distribución porcentual de impactos en la cabeza, el torso y las piernas.
### 3. Imprime todos los contadores y porcentajes.
### 4. Finalmente, utiliza matplotlib para generar y mostrar un gráfico de barras que visualiza claramente la distribución porcentual de los aciertos por zona.
### ESTADÍSTICAS
```
valid_hits = aim_head + aim_torso + aim_legs # Total de aciertos válidos (donde se apuntó a un enemigo)
```
### Calcula el porcentaje total de frames apuntando al enemigo
```
p_total_aim = (aim_on_target / processed_frames) * 100 if processed_frames > 0 else 0

if valid_hits > 0:
    # Calcula el porcentaje de aciertos en cada zona
    p_head  = (aim_head  / valid_hits) * 100
    p_torso = (aim_torso / valid_hits) * 100
    p_legs  = (aim_legs  / valid_hits) * 100
else:
    p_head = p_torso = p_legs = 0
```
### Muestra los resultados en consola
```
print(f"Frames leídos: {total_frames}")
print(f"Frames analizados: {processed_frames}")
print(f"Aiming a enemigo: {aim_on_target}  ({p_total_aim:.2f}%)")

print("\nDistribución REAL de golpes:")
print(f"HEADSHOT:   {aim_head}  ({p_head:.2f}%)")
print(f"BODYSHOT:   {aim_torso} ({p_torso:.2f}%)")
print(f"LEGSHO:   {aim_legs}  ({p_legs:.2f}%)")
```
### GRÁFICO DE BARRAS
```
labels = ["HEADSHOT", "BODYSHOT", "LEGSHOT"]
values = [p_head, p_torso, p_legs]
```
### Crea el gráfico de barras con Matplotlib
```
plt.figure(figsize=(6,4))
plt.bar(labels, values)
plt.title("Aciertos por Zona (%)")
plt.ylabel("Porcentaje")
plt.ylim(0,100) # Establece el eje Y de 0 a 100%
plt.show() # Muestra el gráfico
```
## Tercera parte del código
### Esta sección inicializa el proceso solicitando al usuario nueve valores que representan el porcentaje de aciertos por zona (Headshot, Torso, Piernas) en tres diferentes contextos: un Clip (representando una sola ronda), la Partida Completa y el Episodio Actual (un periodo de tiempo más largo, quizás una temporada o un conjunto de partidas). Una vez ingresados, los datos se organizan en listas (clip_stats, match_stats, epi_stats) para facilitar su uso en las gráficas, y se prepara la estructura numérica (x, width) necesaria para crear un gráfico de barras comparativo.
```
import matplotlib.pyplot as plt # Importa Matplotlib para la creación de gráficos
import numpy as np # Importa NumPy para manejo de arrays y operaciones numéricas (como np.arange)
```
### 1. Solicitar datos al usuario
```
print("=== INGRESA LAS ESTADÍSTICAS DEL CLIP (1 ronda) ===")
```
### Solicita el porcentaje de Headshot para el clip y lo convierte a número flotante
```
clip_head = float(input("Porcentaje Headshot (clip): "))
clip_body = float(input("Porcentaje Torso (clip): "))
clip_legs = float(input("Porcentaje Piernas (clip): "))

print("\n=== INGRESA LAS ESTADÍSTICAS DE LA PARTIDA COMPLETA ===")
match_head = float(input("Porcentaje Headshot (partida): "))
match_body = float(input("Porcentaje Torso (partida): "))
match_legs = float(input("Porcentaje Piernas (partida): "))

print("\n=== INGRESA LAS ESTADÍSTICAS DEL EPISODIO ACTUAL ===")
epi_head = float(input("Porcentaje Headshot (episodio): "))
epi_body = float(input("Porcentaje Torso (episodio): "))
epi_legs = float(input("Porcentaje Piernas (episodio): "))
```
### 2. Preparar datos para la gráfica
```
labels = ["Head", "Torso", "Legs"] # Etiquetas para el eje X
clip_stats = [clip_head, clip_body, clip_legs] # Datos del clip
match_stats = [match_head, match_body, match_legs] # Datos de la partida
epi_stats = [epi_head, epi_body, epi_legs] # Datos del episodio

x = np.arange(len(labels)) # Crea un array de posiciones [0, 1, 2] para el eje X
width = 0.25 # Define el ancho de cada barra para la separación
```
### Esta sección genera la primera gráfica, un diagrama de barras agrupadas que compara directamente las tres fuentes de datos (Clip, Partida, Episodio). Utiliza x - width, x, y x + width para desplazar las barras y agruparlas por zona de impacto (Head, Torso, Legs). Esto permite visualizar rápidamente si el rendimiento del clip (ronda individual) es atípico o si sigue la tendencia general.
### Posteriormente, la sección calcula una estimación proyectada promediando las estadísticas de las tres fuentes para cada zona. Esto crea un valor más robusto y representativo del rendimiento del jugador a largo plazo, asumiendo que el rendimiento futuro se parecerá a la media de su rendimiento reciente.
### 3. Gráfica comparativa
```
plt.figure(figsize=(10,6)) # Crea la figura con un tamaño específico
```
### Primera barra: Clip (desplazada a la izquierda)
```
plt.bar(x - width, clip_stats, width, label='Clip (1 ronda)')
```
### Segunda barra: Partida (en la posición central)
```
plt.bar(x, match_stats, width, label='Partida completa')
```
### Tercera barra: Episodio (desplazada a la derecha)
```
plt.bar(x + width, epi_stats, width, label='Episodio actual')
plt.xticks(x, labels) # Coloca las etiquetas "Head", "Torso", "Legs" bajo los grupos de barras
plt.ylabel("Porcentaje (%)") # Etiqueta del eje Y
plt.title("Comparativa de Precisión por Zonas: Clip vs Partida vs Episodio")
plt.legend() # Muestra la leyenda para identificar cada color
plt.grid(axis='y', linestyle='--', alpha=0.5) # Añade líneas de cuadrícula horizontales
plt.show() # Muestra la gráfica comparativa
```
### 4. Estimación proyectada a 13 rondas
### Calcula el promedio simple de los porcentajes de Headshot de las tres fuentes
```
proj_head = (clip_head + match_head + epi_head) / 3
```
### Calcula el promedio de Torso
```
proj_body = (clip_body + match_body + epi_body) / 3
```
### Calcula el promedio de Piernas
```
proj_legs = (clip_legs + match_legs + epi_legs) / 3

projection = [proj_head, proj_body, proj_legs] # Agrupa los promedios para la segunda gráfica
```
### La sección final crea la segunda gráfica, que visualiza la proyección promedio calculada en el paso anterior. Esta gráfica de barras simple muestra el rendimiento esperado del jugador en una partida de 13 rondas, basándose en el promedio de sus estadísticas. Se utilizan diferentes colores para distinguir las barras (rojo para Head, verde para Torso, azul para Piernas). Finalmente, se imprime un resumen de los porcentajes proyectados en la consola, ofreciendo un resultado numérico limpio de las estimaciones.
### 5. Gráfica de estimación
```
plt.figure(figsize=(8,5)) # Crea la figura con un tamaño para la gráfica de proyección
```
### Genera la gráfica de barras usando los promedios calculados
```
plt.bar(labels, projection, color=['red', 'green', 'blue'])

plt.title("Estimación promedio para una partida de 13 rondas")
plt.ylabel("Porcentaje proyectado (%)")
plt.grid(axis='y', linestyle='--', alpha=0.5) # Añade líneas de cuadrícula horizontales
plt.show() # Muestra la gráfica de estimación
```
### 6. Resumen en consola
```
print("\nESTIMACIÓN DE 13 RONDAS")
```
### Muestra los resultados finales, formateando a dos decimales
```
print(f"Headshot proyectado: {proj_head:.2f}%")
print(f"Torso proyectado: {proj_body:.2f}%")
print(f"Piernas proyectado: {proj_legs:.2f}%")
```
## El conjunto de código que desarrollamos representa un sistema completo para el análisis de rendimiento y precisión de puntería (aiming) diseñado específicamente para clips de videojuegos en primera persona (FPS). En la primera etapa, establecimos el procesamiento básico del video, mientras que la segunda etapa nos permitió implementar la detección de objetos con YOLOv8 para identificar a los enemigos ("person"). El aspecto más importante es cómo utilizamos el centro exacto de la pantalla como la mira del jugador para evaluar la precisión. Clasificamos cada impacto como Headshot, Bodyshot o Legshot basándome en la segmentación de la caja delimitadora del enemigo. El objetivo final, reflejado en la tercera parte, fue ir más allá del análisis del clip individual; por ello, integramos las estadísticas del clip con los datos de rendimiento a nivel de Partida y Episodio. Al generar gráficos de barras comparativos y cálculos de proyección promedio, logramos ofrecer una evaluación objetiva y visualmente clara de mi rendimiento y tendencias de precisión a lo largo del tiempo, lo cual nos ayuda a identificar específicamente dónde debemos mejorar la puntería en ese entorno de juego en primera persona.
