from face_recognizer import FaceRecognizer, speak
import cv2
import matplotlib.pyplot as plt

img_path = 'Mohamed_Salah_2.jpg'

img = cv2.imread(img_path)
fce = FaceRecognizer()

img, name = fce.get_faces(img)

print(name)

plt.imshow(img)
plt.axis("off")
plt.show()

speak(name)
