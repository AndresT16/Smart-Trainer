import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calcular_angulo(a,b,c):
    a = np.array(a) # Primero
    b = np.array(b) # Medio
    c = np.array(c) # Final
    radianes = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angulo = np.abs(radianes * 180.0 / np.pi)

    if angulo > 180.0:
        angulo = 360 - angulo
    return angulo

# transmision de video
captura = cv2.VideoCapture(0)

# Variables del contador de flexor
contador = 0 
estado = None

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

        #Extraccion puntos de referencia
        try:
            landmarks = resultado.pose_landmarks.landmark

            # Obtener coordenadas
            hombro = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            codo = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            munieca = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            # Calcular ángulo
            angulo = calcular_angulo(hombro, codo, munieca)

            # Visualizar ángulo
            cv2.putText(imagen, str(angulo),
                        tuple(np.multiply(codo, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            
            # logica del contador de flexor de bicep
            if angulo > 160:
                estado = "abajo"
            if angulo < 30 and estado =='abajo':
                estado="arriba"
                contador +=1
                print(contador)

        except:
            pass
            
             # Mostrar por pantalla los datos 
        # Planteando un rectangulo
        cv2.rectangle(image, (0,0), (225,73), (0,128,0), -1)
        
        # Dato Repeticion
        cv2.putText(image, 'REP', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(contador), 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
        
        # Dato Estado
        cv2.putText(image, 'ESTADO', (65,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, estado, 
                    (60,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

        
        #Deteccion de renderizado 
        mp_dibujo.draw_landmarks(imagen,resultado.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                 mp_dibujo.DrawingSpec(color=(245,117,66), thickness= 2, circle_radius=2),
                                 mp_dibujo.DrawingSpec(color=(245,66,230), thickness= 2, circle_radius=2))
        cv2.imshow("Mediapipe Feed",imagen)
        if cv2.waitKey(10) & 0xFF == ord("x"):
            break
captura.release()
cv2.destroyAllWindows()    
