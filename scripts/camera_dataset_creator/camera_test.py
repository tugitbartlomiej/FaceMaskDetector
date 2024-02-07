import cv2

# Inicjalizacja detektora twarzy
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Nie można otworzyć kamery")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

desired_positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
positions_found = [False, False, False, False]
face_found_in_all_quadrants = False

while True:
    # Odczyt jednej ramki
    ret, frame = cap.read()
    if not ret:
        print("Nie można odczytać ramki. Kończenie...")
        break

    out.write(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    height, width = frame.shape[:2]
    mid_x, mid_y = width // 2, height // 2

    cv2.line(frame, (mid_x, 0), (mid_x, height), (255, 0, 0), 2)
    cv2.line(frame, (0, mid_y), (width, mid_y), (255, 0, 0), 2)

    for (x, y, w, h) in faces:
        for i, (quadrant_x, quadrant_y) in enumerate(desired_positions):
            if quadrant_x * mid_x <= x + w//2 <= (quadrant_x + 1) * mid_x and \
               quadrant_y * mid_y <= y + h//2 <= (quadrant_y + 1) * mid_y:
                positions_found[i] = True
                # Rysowanie prostokąta wokół twarzy
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    if all(positions_found) and not face_found_in_all_quadrants:
        face_found_in_all_quadrants = True
        print("Twarz została wykryta we wszystkich kwadrantach.")
        cv2.putText(frame, "Twarz znaleziona we wszystkich kwadrantach!", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Wyświetlenie wyników
    cv2.imshow('Kamera', frame)

    # Wyjście z pętli, jeśli naciśnięto klawisz 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zwolnienie kamery i zamknięcie okien
cap.release()
out.release()
cv2.destroyAllWindows()