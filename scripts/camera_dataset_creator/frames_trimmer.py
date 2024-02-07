import cv2
import os

input_video = 'Mask.mp4'
# input_video = 'Mask.mp4'

output_folder = 'movie_frames'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("Error can not open video.")
    exit()

frame_number = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Save only every 10th
    if frame_number % 10 == 0:
        frame_path = os.path.join(output_folder, f"frame{frame_number:04d}.png")
        cv2.imwrite(frame_path, frame)

    frame_number += 1


cap.release()
cv2.destroyAllWindows()

print(f"Saved {frame_number} frames.")
