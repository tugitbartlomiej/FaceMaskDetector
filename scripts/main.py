import cv2
print(cv2.__version__)

cap = cv2.VideoCapture(0)

if not (cap.isOpened()):

    print('Could not open video device')

#To set the resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while(True):
    # Capture frame-by-frame

    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('preview', frame)

    #Waits for a user input to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()