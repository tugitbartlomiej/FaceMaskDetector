import cv2
import os

def save_face(face, frame, folder):
    (x, y, w, h) = face
    face_image = frame[y:y+h, x:x+w]
    cv2.imwrite(f"{folder}/{cv2.getTickCount()}.png", face_image)

def is_face_in_region(face, region):
    (x, y, w, h) = face
    (rx, ry, rw, rh) = region
    return x >= rx and y >= ry and x + w <= rx + rw and y + h <= ry + rh


def create_folders(base_folder, subfolders):
    for folder in subfolders:
        os.makedirs(os.path.join(base_folder, folder), exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(2)

# Tworzenie folderów do zapisu zdjęć
base_folder = "captured_faces"
subfolders = ["top_left", "top_right", "bottom_left", "bottom_right"]
create_folders(base_folder, subfolders)

frame = cap.read()
frame = cap.read()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        print("Wykryto twarz!")

    height, width = frame.shape[:2]
    regions = {
        "top_left": (0, 0, width // 2, height // 2),
        "top_right": (width // 2, 0, width // 2, height // 2),
        "bottom_left": (0, height // 2, width // 2, height // 2),
        "bottom_right": (width // 2, height // 2, width // 2, height // 2)
    }

    cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 0, 0), 2)  # Niebieska linia pionowa
    cv2.line(frame, (0, height // 2), (width, height // 2), (255, 0, 0), 2)  # Niebieska linia pozioma

    for face in faces:
        for region_name, region in regions.items():
            if is_face_in_region(face, region):
                save_face(face, frame, os.path.join(base_folder, region_name))

    cv2.imshow('Frame with Lines', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
