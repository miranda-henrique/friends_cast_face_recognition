import cv2
import numpy as np
import face_recognition

# load images
img = cv2.imread("./source_code/Messi1.png")
img_in_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(img_in_rgb)[0]

img2 = cv2.imread("./source_code/images/messi.png")
img2_in_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2_encoding = face_recognition.face_encodings(img2_in_rgb)[0]

# plt.imshow(img2_in_rgb)
# plt.show()

# check if images contain the same person
result = face_recognition.compare_faces(np.array([img_encoding]), np.array([img2_encoding]))
print("Result: ", result)