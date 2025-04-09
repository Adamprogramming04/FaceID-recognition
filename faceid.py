import cv2
import os
import numpy as np
import webbrowser
from flask import Flask

# === SETUP FLASK ===
app = Flask(__name__)

@app.route('/')
def welcome():
    return "<h1 style='text-align:center;margin-top:100px;'> Welcome!</h1>"

# === LAUNCH FLASK ===
def launch_flask():
    from threading import Thread
    import socket

    ip = socket.gethostbyname(socket.gethostname())
    url = f"http://{ip}:5000"
    print(f"🌐 Access Flask app at {url}")
    webbrowser.open(url)

    Thread(target=app.run, kwargs={'host': '0.0.0.0', 'debug': False, 'use_reloader': False}).start()

# === FACE SETUP ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
TRAINING_FILE = "trained_face.yml"
SAVED_FACE_IMG = "saved_face.jpg"

# === REGISTER FACE ===
def register_face():
    cap = cv2.VideoCapture(0)
    print("📸 Press 's' to save your face")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        cv2.imshow("Register Face", frame)

        key = cv2.waitKey(1)
        if key == ord('s') and len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))
            cv2.imwrite(SAVED_FACE_IMG, face_img)
            face_recognizer.train([face_img], np.array([0]))
            face_recognizer.save(TRAINING_FILE)
            print(" Face registered!")
            break

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# === RECOGNIZE FACE ===
def recognize_face():
    if not os.path.exists(TRAINING_FILE):
        print(" No face registered. Run option 1 first.")
        return

    face_recognizer.read(TRAINING_FILE)
    cap = cv2.VideoCapture(0)
    print("🔍 Recognizing... Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))

            label, confidence = face_recognizer.predict(face_img)
            print(f"Confidence: {confidence:.2f}")

            if confidence < 50:  # Lower = better
                print(" Face recognized! Launching Flask.")
                cap.release()
                cv2.destroyAllWindows()
                launch_flask()
                return

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# === MAIN MENU ===
if __name__ == "__main__":
    print("""
============================
  FACE ID AUTHENTICATION
============================
[1] Register Face
[2] Start Face Recognition
""")
    choice = input("Enter choice (1 or 2): ")

    if choice == "1":
        register_face()
    elif choice == "2":
        recognize_face()
    else:
        print(" Invalid choice.")
