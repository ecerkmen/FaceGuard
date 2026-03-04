import face_recognition
import os
import pickle
from sklearn.svm import SVC

encodings = []
labels = []

dataset_path = os.path.join(os.path.expanduser("~"), "Desktop", "dataset")

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if os.path.isdir(person_folder):
        for image in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image)
            loaded_image = face_recognition.load_image_file(image_path)
            face_enc = face_recognition.face_encodings(loaded_image)
            if face_enc:
                encodings.append(face_enc[0])
                labels.append(person_name)

print(f"Found {len(encodings)} encodings")
classifier = SVC(kernel="rbf", probability=True)
classifier.fit(encodings, labels)

with open("model.pkl", "wb") as f:
    pickle.dump(classifier, f)