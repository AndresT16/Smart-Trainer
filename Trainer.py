import cv2
import mediapipe as mp
import numpy as np
import time
import asyncio
import tkinter as tk
from tkinter import simpledialog
from telegram import Bot

TOKEN = "6410070119:AAFLGiHvy3-Qx-adMkhFHY6XPDMnvdHnAYs"
CHAT_ID = "-1002128687767"
bot = Bot(token=TOKEN)

# Crear una nueva ventana de Tkinter
ROOT = tk.Tk()
# Ocultar la ventana principal de Tkinter
ROOT.withdraw()
# Solicitar el nombre de usuario a través de una ventana de diálogo
usuario = simpledialog.askstring(title="Smart Trainer",
                                  prompt="Por favor, introduce tu usuario de Telegram:")


# Cargar el clasificador pre-entrenado de OpenCV para la detección de rostros
cascada = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pTime = 0

def calcular_angulo(a, b, c):
    a = np.array(a)  # Inicio
    b = np.array(b)  # Medio
    c = np.array(c)  # Fin
    radianes = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angulo = np.abs(radianes * 180.0 / np.pi)
    if angulo > 180.0:
        angulo = 360 - angulo
    return angulo

# Solicitar al usuario que introduzca el índice de la cámara
indice_camara = simpledialog.askinteger(title="Smart Trainer",
                                        prompt="Por favor, introduce el índice de la cámara que deseas usar:")

# Utilizar el índice de la cámara proporcionado por el usuario
cap = cv2.VideoCapture(indice_camara)

# Variables del contador de flexor
contador = 0
estado = None

# fps variable
fps = 0

## Configurar instancia de Mediapipe
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Convertir la imagen a escala de grises para la detección de rostros
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostros
        rostros = cascada.detectMultiScale(gray, 1.1, 4)

        # Para cada rostro detectado, desenfoca el rostro en la imagen
        for (x, y, w, h) in rostros:
            face = frame[y:y+h, x:x+w]
            # Aplica un desenfoque a la región del rostro
            blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
            # Sustituye la región del rostro en la imagen original por la versión desenfocada
            frame[y:y+h, x:x+w] = blurred_face

        # Recolorar imagen a RGB
        imagen = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagen.flags.writeable = False

        # Realizar detección
        resultados = pose.process(imagen)

        # Recolorar de nuevo a BGR
        imagen.flags.writeable = True
        imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)

        # Extraer puntos de referencia
        try:
            landmarks = resultados.pose_landmarks.landmark

            # Obtener coordenadas
            hombro = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            codo = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            muneca = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calcular ángulo
            angulo = calcular_angulo(hombro, codo, muneca)

            # Visualizar ángulo
            cv2.putText(imagen, str(angulo),
                        tuple(np.multiply(codo, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )

            # Lógica del contador de flexor de bíceps
            if angulo > 160:
                estado = "abajo"
            if angulo < 30 and estado == 'abajo':
                estado = "arriba"
                contador += 1
                print(contador)

        except Exception as e:
            print(f"Error: {e}")

        # Mostrar por pantalla los datos
        # Planteando un rectángulo
        cv2.rectangle(imagen, (0, 0), (225, 73), (245, 117, 66), -1)

        # Dato Repetición
        cv2.putText(imagen, 'REP', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(imagen, str(contador),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Dato Estado
        cv2.putText(imagen, 'ESTADO', (65, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(imagen, estado,
                    (60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(imagen, f'FPS: {int(fps)}',
                    (500, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Detección de renderizado
        mp_drawing.draw_landmarks(imagen, resultados.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Smart Trainer', imagen)

        if cv2.waitKey(10) & 0xFF == ord('x'):
            break
        # Calcular fps
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

    Mensaje_bot = f"Felicitaciones @{usuario} en esta sesión realizaste un total de {contador} repeticiones, ¡Sigue así!"


    # Define una función asíncrona para enviar el mensaje
    async def send_message():
        await bot.send_message(chat_id=CHAT_ID, text=Mensaje_bot)

    # Usa asyncio para ejecutar la función send_message
    asyncio.run(send_message())

    cap.release()
    cv2.destroyAllWindows()
