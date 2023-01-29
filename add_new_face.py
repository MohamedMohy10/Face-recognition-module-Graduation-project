from face_recognizer import FaceRecognizer
import cv2

fce = FaceRecognizer()
# path to the image of face to be added
camera = cv2.imread('Mohamed_Salah_2018.jpg')
camera = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
# name of the person to be added
fce.add_new_face(camera, 'Mohamed Salah')
