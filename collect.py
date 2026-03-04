import cv2
import face_recognition
import os

name = input("Enter the name for this person: ")

os.makedirs(name, exist_ok=True)
camera = cv2.VideoCapture(0)

counter = 0

while camera.isOpened():
    ret, frame = camera.read()

    if not ret:
        continue

    cv2.imshow("Collecting Faces", frame)
    key = cv2.waitKey(1)
    
    if key == ord("q"):
        break
    if key == 32:
        cv2.imwrite(f"{name}/{counter}.jpg", frame)
        counter += 1
        print(f"Saved photo {counter}")
        if counter == 30:
            break

camera.release()
cv2.destroyAllWindows()