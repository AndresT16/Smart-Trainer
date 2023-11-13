import cv2
import mediapipe as mp
import numpy as np

mp_dibujo = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# transmision de video
captura = cv2.VideoCapture(0)
#Configurar mediapipe instancia 
with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    while captura.isOpened():
        retorno, frame = captura.read()
        #Recolorear imagen a RGB
        imagen = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        imagen.flags.writeable = False
        #Realizar detecciones
        resultado = pose.process(imagen)
        #Recolorear de nuevo a BGR
        imagen.flags.writeable = True
        imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
        #Deteccion de renderizado 
        mp_dibujo.draw_landmarks(imagen,resultado.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                 mp_dibujo.DrawingSpec(color=(245,117,66), thickness= 2, circle_radius=2),
                                 mp_dibujo.DrawingSpec(color=(245,117,66), thickness= 2, circle_radius=2))
        cv2.imshow("Mediapipe Feed",frame)
        if cv2.waitKey(10) & 0xFF == ord("s"):
            break
captura.release()
cv2.destroyAllWindows()    
