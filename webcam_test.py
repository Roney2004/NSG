import pickle
import numpy as np
import torch
import cv2
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image

# ------------------ Load Stored Model ------------------
with open("facerec_model.pkl", "rb") as f:
    face_model_list = pickle.load(f)

print("Loaded face entries:", len(face_model_list))

# ------------------ Initialize Models ------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ------------------ Cosine Similarity ------------------
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ------------------ Start Webcam ------------------
cap = cv2.VideoCapture(0)

threshold = 0.6

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    face = mtcnn(img_pil)

    if face is not None:
        face = face.to(device)

        with torch.no_grad():
            embedding = resnet(face.unsqueeze(0)).cpu().numpy()[0]

        best_score = -1
        best_name = "Unknown"

        for entry in face_model_list:
            name = entry["identity"].split("/")[-2]
            stored_embedding = np.array(entry["embedding"])
            sim = cosine_sim(embedding, stored_embedding)

            if sim > best_score:
                best_score = sim
                best_name = name

        if best_score < threshold:
            best_name = "Unknown"

        cv2.putText(frame, f"{best_name} ({best_score:.2f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()