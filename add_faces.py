import cv2
import pickle
import numpy as np
import os

if not os.path.exists('data/'):
    os.makedirs('data/')

cap = cv2.VideoCapture(0)  # Use default camera

facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces_data = []
i = 0
name = input("Enter your Aadhar number: ")
framesTotal = 51
captureAfterFrame = 2

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (50, 50))
        if len(faces_data) < framesTotal and i % captureAfterFrame == 0:
            faces_data.append(resized_img)
        i += 1
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) >= framesTotal:
        break

cap.release()
cv2.destroyAllWindows()

faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape((framesTotal, -1))  # flatten each face to 1D

# Load or create names list
names_path = 'data/names.pkl'
faces_path = 'data/faces_data.pkl'

if not os.path.isfile(names_path):
    names = [name] * framesTotal
else:
    with open(names_path, 'rb') as f:
        names = pickle.load(f)
    names += [name] * framesTotal

with open(names_path, 'wb') as f:
    pickle.dump(names, f)

# Load or create faces_data array
if not os.path.isfile(faces_path):
    faces = faces_data
else:
    with open(faces_path, 'rb') as f:
        faces = pickle.load(f)
    if faces.shape[1] == faces_data.shape[1]:
        faces = np.append(faces, faces_data, axis=0)
    else:
        print(f"Shape mismatch! Existing faces shape: {faces.shape}, New faces shape: {faces_data.shape}")
        exit(1)

with open(faces_path, 'wb') as f:
    pickle.dump(faces, f)

print(f"Data saved successfully! Total faces now: {len(faces)}")
