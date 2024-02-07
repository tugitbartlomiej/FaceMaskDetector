import cv2
import os
import shutil
import numpy as np

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, image

    # Rozpoczęcie rysowania prostokąta
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    # Aktualizacja wymiarów prostokąta podczas ruchu myszki
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            image2 = image.copy()
            cv2.rectangle(image2, (ix, iy), (x, y), (255, 0, 0), 2)
            cv2.imshow('image', image2)

    # Zakończenie rysowania prostokąta
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(image, (ix, iy), (x, y), (255, 0, 0), 2)
        save_annotation(ix, iy, x, y)

def save_annotation(x1, y1, x2, y2):
    # Obliczenie i zapisanie współrzędnych w formacie YOLOv8
    label = 0  # ID klasy (domyślnie 0, zmień jeśli masz więcej klas)
    x_center = round((x1 + x2) / (2 * width), 6)
    y_center = round((y1 + y2) / (2 * height), 6)
    bbox_width = round(abs(x2 - x1) / width, 6)
    bbox_height = round(abs(y2 - y1) / height, 6)

    with open(label_path, 'a') as file:
        file.write(f"{label} {x_center} {y_center} {bbox_width} {bbox_height}\n")

def process_image():
    # Skopiowanie zdjęcia do folderu images i usunięcie z folderu not_detected_images
    images_dir_path = os.path.join(images_dir, image_name)
    shutil.copy(image_path, images_dir_path)
    os.remove(image_path)


def resize_and_center_image(img, screen_res):
    img_h, img_w = img.shape[:2]
    screen_w, screen_h = screen_res

    # Proporcje skalowania, aby obraz pasował do ekranu
    scale = min(screen_w/img_w, screen_h/img_h)

    # Nowe wymiary obrazu
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    # Skalowanie obrazu
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Tworzenie nowego obrazu o rozmiarze ekranu
    centered_img = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

    # Wyznaczenie pozycji wstawienia skalowanego obrazu
    top_left_x = (screen_w - new_w) // 2
    top_left_y = (screen_h - new_h) // 2

    # Wstawienie obrazu na środku nowego obrazu
    centered_img[top_left_y:top_left_y+new_h, top_left_x:top_left_x+new_w] = resized_img

    return centered_img

not_detected_dir = '/home/bartlomiej/Studia/Sem4/Przetwarzanie_Obrazów/face-masks/scripts/yolov8/dataset/not_detected/not_detected_images'
label_dir = 'dataset/not_detected/labels'
images_dir = 'dataset/not_detected/images'
os.makedirs(label_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)


cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
screen_res = (1920, 1080)  # Ustaw rozmiar ekranu

for image_name in os.listdir(not_detected_dir):
    image_path = os.path.join(not_detected_dir, image_name)
    label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + '.txt')

    # Pierwsze wczytanie obrazu
    image = cv2.imread(image_path)
    original_image = image.copy()  # Zapisanie kopii oryginalnego obrazu
    screen_size = (1920, 1080)  # Ustaw rozmiar ekranu
    image = resize_and_center_image(image, screen_size)
    height, width = image.shape[:2]
    drawing = False
    ix, iy = -1, -1

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_rectangle)

    while True:
        cv2.imshow('image', image)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('n'):  # Naciśnij 'n', aby przejść do następnego obrazu i przetworzyć aktualne zdjęcie
            process_image()
            break
        elif k == ord(' '):  # Naciśnij spację, aby przejść do następnego obrazu bez przetwarzania
            break
        elif k == ord('r'):  # Naciśnij 'r', aby odświeżyć bieżące zdjęcie
            image = original_image.copy()  # Wczytanie oryginalnego obrazu bez zaznaczeń
        elif k == ord('q'):  # Naciśnij 'q', aby wyjść
            cv2.destroyAllWindows()
            exit()

    cv2.destroyAllWindows()

print('Process completed.')
