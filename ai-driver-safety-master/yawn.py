import cv2
import dlib
import numpy as np


PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat" #modèle de prédiction des repères faciaux 
predictor = dlib.shape_predictor(PREDICTOR_PATH)
#cascade_path='haarcascade_frontalface_default.xml'
#cascade = cv2.CascadeClassifier(cascade_path)
detector = dlib.get_frontal_face_detector() # la fonction-->détecteur de visages


def get_landmarks(im):#Utilise le détecteur pour trouver un visage dans l'image im ET utilise le prédicteur pour extraire les repères faciaux et Retourne les coordonnées des repères
    rects = detector(im, 1)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):#Dessine les repères sur l'image im et renvoie l'image modifiée
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def top_lip(landmarks): #Extrait les repères du haut des lèvres et calcule leur centre moyen
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append(landmarks[i])
    for i in range(61,64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:,1])

def bottom_lip(landmarks): #Extrait les repères du bas des lèvres et calcule leur centre moyen.
    bottom_lip_pts = []
    for i in range(65,68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56,59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:,1])

def mouth_open(image): #pour détecter l'ouverture de la bouche
    landmarks = get_landmarks(image) #landmarks est censé contenir les coordonnées des repères faciaux détectés sous forme d'une matrice NumPy
    
    if isinstance(landmarks, str) and landmarks == "error":# erreur s'est produite lors de la détection des repères faciaux
        return image, 0 #retourne l'image d'origine (non modifiée) et 0 pour la distance entre les lèvres
    
    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center) # Calcule la distance verticale entre les lèvres
    return image_with_landmarks, lip_distance

    #cv2.imshow('Result', image_with_landmarks)
    #cv2.imwrite('image_with_landmarks.jpg',image_with_landmarks)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)
yawns = 0
yawn_status = False 

while True: #boucle principale
    ret, frame = cap.read()   
    image_landmarks, lip_distance = mouth_open(frame)
    
    prev_yawn_status = yawn_status  
    
    if lip_distance > 40:
        yawn_status = True 
        
        cv2.putText(frame, "Subject is Yawning", (50,450), 
                    cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
        

        output_text = " Yawn Count: " + str(yawns + 1) #Affiche le nombre de bâillements détectés en temps réel sur la vidéo de la webcam

        cv2.putText(frame, output_text, (50,50),
                    cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
        
    else:
        yawn_status = False 
         
    if prev_yawn_status == True and yawn_status == False:
        yawns += 1
 
    cv2.imshow('Live Landmarks', image_landmarks ) #fenêtre affiche la vidéo de la webcam avec les repères faciaux annotés en temps réel. Les repères faciaux sont dessinés sur le visage détecté.
    cv2.imshow('Yawn Detection', frame )#fenêtre affiche également la vidéo de la webcam, mais avec des annotations indiquant si un bâillement est détecté. Si un bâillement est détecté, un texte "Subject is Yawning" est affiché, ainsi qu'un compteur de bâillements.
    
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows() 
