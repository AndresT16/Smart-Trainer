import cv2
import mediapipe as mp
import numpy as np

dibujo = mp.solutions.drawing_utils
deteccion_pose = mp.solutions.pose

# transmision de video
captura = cv2.VideoCapture(0)
while captura.isOpened():
    retorno , frame = captura.read()
    cv2.imshow("Mediapipe Feed",frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

captura.release()
cv2.destroyAllWindows()    