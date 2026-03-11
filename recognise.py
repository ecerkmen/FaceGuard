import asyncio
import cv2
import face_recognition
import io
import numpy
import os
import pickle
import time
from collections import Counter
from dotenv import load_dotenv
from telegram import Bot

load_dotenv()

with open("model.pkl", "rb") as f:
    classifier = pickle.load(f)

camera = cv2.VideoCapture(0)

bot = Bot(token=os.getenv("TELEGRAM_TOKEN"))

def blur_score(face_crop):
    grey = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(grey, cv2.CV_64F).var()
    return score

tracked_faces = {}
face_counter = 0

while camera.isOpened():
    ret, frame = camera.read()

    if not ret:
        continue

    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    face_locs = face_recognition.face_locations(small_frame)
    face_encs = face_recognition.face_encodings(small_frame, face_locs)

    for face_enc, face_loc in zip(face_encs, face_locs):
        matched_id = None
        probabilities = classifier.predict_proba([face_enc])[0]
        confidence = max(probabilities)
        name = classifier.classes_[probabilities.argmax()]

        if confidence < 0.8:
            name = "Unknown person"
        print(f"Detecting: {name}, Confidence: {confidence}")
       
        top, right, bottom, left = [coord * 4 for coord in face_loc]
        face_crop = frame[top:bottom, left:right]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top -10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        for face_id in tracked_faces:
            distance = face_recognition.face_distance([tracked_faces[face_id]["encoding"]], face_enc)[0]
            if distance < 0.6:
                matched_id = face_id
                break
        if matched_id is None:
            tracked_faces[face_counter] = {
                "encoding": face_enc,
                "frames": [],
                "last_capture": 0
            }
            face_counter += 1
        else:
            frame_data = {
                "image": frame,
                "confidence": max(probabilities),
                "blur_score": blur_score(face_crop),
                "name": name
            }
            if time.time() - tracked_faces[matched_id]["last_capture"] > 1:
                tracked_faces[matched_id]["last_capture"] = time.time()
                tracked_faces[matched_id]["frames"].append(frame_data)

            if len(tracked_faces[matched_id]["frames"]) == 10:
                tracked_faces[matched_id]["frames"].sort(key=lambda x: x["blur_score"])
                del tracked_faces[matched_id]["frames"][0:5]
                confidence_calc = [d.get("confidence") for d in tracked_faces[matched_id]["frames"]]
                final_confidence = sum(confidence_calc) / 5
                all_names = [d["name"] for d in tracked_faces[matched_id]["frames"]]
                name_counter = Counter(all_names)
                freq_name = name_counter.most_common(1)[0][0]

                if freq_name != "Ece":
                    ret, buffer = cv2.imencode(".jpeg", tracked_faces[matched_id]["frames"][4]["image"])
                    image_bytes = io.BytesIO(buffer.tobytes())
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(bot.send_message(chat_id=int(os.getenv("TELEGRAM_CHAT_ID")), text=f"{freq_name} is at the door."))
                    loop.run_until_complete(bot.send_photo(chat_id=int(os.getenv("TELEGRAM_CHAT_ID")), photo = image_bytes))
                tracked_faces[matched_id]["frames"] = []
            
    cv2.imshow("Facial Recognition", frame)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()