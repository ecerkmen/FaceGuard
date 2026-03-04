import asyncio
import cv2
import face_recognition
import io
import numpy
import os
import pickle
import time
from dotenv import load_dotenv
from telegram import Bot

load_dotenv()

with open("model.pkl", "rb") as f:
    classifier = pickle.load(f)

camera = cv2.VideoCapture(0)

notified = set()
bot = Bot(token=os.getenv("TELEGRAM_TOKEN"))

while camera.isOpened():
    ret, frame = camera.read()

    if not ret:
        continue

    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    face_locs = face_recognition.face_locations(small_frame)
    face_encs = face_recognition.face_encodings(small_frame, face_locs)
    for face_enc, face_loc in zip(face_encs, face_locs):
        probabilities = classifier.predict_proba([face_enc])[0]
        confidence = max(probabilities)
        name = classifier.classes_[probabilities.argmax()]
        if confidence < 0.7:
            name = "Unknown person"
        print(f"Detecting: {name}, Confidence: {confidence}")
        if name not in notified and name != "Ece":
            ret, buffer = cv2.imencode(".jpeg", frame)
            image_bytes = io.BytesIO(buffer.tobytes())
            loop = asyncio.get_event_loop()
            loop.run_until_complete(bot.send_message(chat_id=int(os.getenv("TELEGRAM_CHAT_ID")), text=f"{name} is at the door."))
            loop.run_until_complete(bot.send_photo(chat_id=int(os.getenv("TELEGRAM_CHAT_ID")), photo=image_bytes))
            notified.add(name)
        top, right, bottom, left = [coord * 4 for coord in face_loc]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top -10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
    cv2.imshow("Facial Recognition", frame)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()