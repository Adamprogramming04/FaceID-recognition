# Face ID Authentication with Flask

This project implements a simple Face ID authentication system using **OpenCV**, **Flask**, and a **web browser** interface. It allows users to register their face, and upon recognition, it opens a Flask web application as a successful login response.

---

## Features

- **Face Registration**: The user can capture and save their face image using a webcam, and it is saved in a trained file for later recognition.
- **Face Recognition**: The system can detect and recognize a registered face using **LBPH (Local Binary Patterns Histograms)** face recognition.
- **Flask Web Application**: When a recognized face is detected, a Flask-based web application is opened in a web browser with a "Welcome!" message.
- **Simple CLI Menu**: A menu-driven interface to either register a face or start face recognition.

---

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- Flask

You can install the required dependencies using pip:

```bash
pip install opencv-python opencv-python-headless flask
