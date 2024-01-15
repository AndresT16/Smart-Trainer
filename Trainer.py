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

ROOT = tk.Tk()
ROOT.withdraw()
usuario = simpledialog.askstring(title="Smart Trainer",
                                  prompt="Por favor, introduce tu usuario de Telegram:")

cascada = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pTime = 0

ANGULO_SUPERIOR = 160
ANGULO_INFERIOR = 30

def calcular_angulo(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radianes = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angulo = np.abs(radianes * 180.0 / np.pi)
    if angulo > 180.0:
        angulo = 360 - angulo
    return angulo

indice_camara = simpledialog.askinteger(title="Smart Trainer",
                                        prompt="Por favor, introduce el índice de la cámara que deseas usar:")

cap = cv2.VideoCapture(indice_camara)

contador = 0
estado = None

fps = 0

def obtener_coordenadas(landmarks, landmark):
    return [landmarks[landmark.value].x, landmarks[landmark.value].y]

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rostros = cascada.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in rostros:
            face = frame[y:y+h, x:x+w]
            blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
            frame[y:y+h, x:x+w] = blurred_face

        imagen = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagen.flags.writeable = False

        resultados = pose.process(imagen)

        imagen.flags.writeable = True
        imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)

        try:
            landmarks = resultados.pose_landmarks.landmark

            hombro = obtener_coordenadas(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
            codo = obtener_coordenadas(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW)
            muneca = obtener_coordenadas(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)

            angulo = calcular_angulo(hombro, codo, muneca)

            if angulo > ANGULO_SUPERIOR:
                estado = "abajo"
            if angulo < ANGULO_INFERIOR and estado == 'abajo':
                estado = "arriba"
                contador += 1
                print(contador)

        except Exception as e:
            print(f"Error: {e}")

        # Mostrar por pantalla los datos
        cv2.rectangle(imagen, (0, 0), (225, 73), (245, 117, 66), -1)

        cv2.putText(imagen, 'REP', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(imagen, str(contador),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(imagen, 'ESTADO', (65, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(imagen, estado,
                    (60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(imagen, f'FPS: {int(fps)}',
                    (500, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        mp_drawing.draw_landmarks(imagen, resultados.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Smart Trainer', imagen)

        if cv2.waitKey(10) & 0xFF == ord('x'):
            break

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

    Mensaje_bot = f"Felicitaciones @{usuario} en esta sesión realizaste un total de {contador} repeticiones, ¡Sigue así!"

    async def send_message():
        await bot.send_message(chat_id=CHAT_ID, text=Mensaje_bot)

    asyncio.run(send_message())

    cap.release()
    cv2.destroyAllWindows()
