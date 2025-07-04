from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
import subprocess

def speak(text):
    subprocess.call(['say', text])

cap = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if not os.path.exists('data/'):
    os.makedirs('data/')

# Load labels and faces data
with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)

with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Check for mismatch in lengths and fix if possible
if len(FACES) != len(LABELS):
    print(f"[ERROR] Faces ({len(FACES)}) and Labels ({len(LABELS)}) count mismatch!")
    exit()

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

imgBackground = cv2.imread("background.jpg")
if imgBackground is None:
    print("[ERROR] background.jpg not found.")
    exit()

# Get background dimensions
bg_height, bg_width = imgBackground.shape[:2]

COL_NAMES = ['NAMES', 'VOTE', 'DATE', 'TIME']

def check_if_exists(value):
    if not os.path.isfile("Votes.csv"):
        return False
    try:
        with open("Votes.csv", "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row and row[0] == value:
                    return True
    except Exception as e:
        print(f"Error reading Votes.csv: {e}")
    return False

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame from camera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    resized_frame = cv2.resize(frame, (640, 432))
    fh, fw = resized_frame.shape[:2]
    start_y = bg_height - fh  # Align to bottom of background
    start_x = 255             # Same X position as before

    # Avoid out-of-bounds error
    if start_y >= 0 and (start_x + fw <= bg_width):
        imgBackground[start_y:start_y+fh, start_x:start_x+fw] = resized_frame

    cv2.imshow('frame', imgBackground)

    if len(faces) == 0:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H-%M-%S")
        exist = os.path.isfile("Votes.csv")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        voter_id = output[0]

        resized_frame = cv2.resize(frame, (640, 432))
        fh, fw = resized_frame.shape[:2]
        start_y = bg_height - fh
        if start_y >= 0 and (start_x + fw <= bg_width):
            imgBackground[start_y:start_y+fh, start_x:start_x+fw] = resized_frame

        cv2.imshow('frame', imgBackground)

        if check_if_exists(voter_id):
            print("YOU HAVE ALREADY VOTED")
            speak("YOU HAVE ALREADY VOTED")
            time.sleep(3)
            cap.release()
            cv2.destroyAllWindows()
            exit()

        k = cv2.waitKey(1) & 0xFF
        if k == ord('1'):
            vote = "BJP"
        elif k == ord('2'):
            vote = "CONGRESS"
        elif k == ord('3'):
            vote = "AAP"
        elif k == ord('4'):
            vote = "NOTA"
        else:
            vote = None

        if vote is not None:
            print(f"YOUR VOTE FOR {vote} HAS BEEN RECORDED")
            speak(f"YOUR VOTE FOR {vote} HAS BEEN RECORDED")
            time.sleep(3)

            if exist:
                with open("Votes.csv", "a", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([voter_id, vote, date, timestamp])
            else:
                with open("Votes.csv", "w", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(COL_NAMES)
                    writer.writerow([voter_id, vote, date, timestamp])

            speak("THANK YOU FOR PARTICIPATING IN THE ELECTIONS")
            break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
