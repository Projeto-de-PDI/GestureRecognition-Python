import cv2
import numpy as np
import base64
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import json
from tensorflow.keras.models import load_model
import mediapipe as mp
import time

# Inicializa o módulo de detecção de mãos do MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Carrega o modelo treinado
model = load_model('models/hand_gesture_landmarks_model.h5')

# Mapeamento das classes
class_names = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10'}

# Dicionário de exercícios (sequência de gestos -> nome do exercício)
exercises = {
    ('0', '1', '0'): 'Exercício 1',  
    ('0', '2', '3', '4'): 'Exercício 2',
    ('0', '5', '0'): 'Exercício 3',           
    ('0', '6', '0'): 'Exercício 4',  
    ('0', '7', '8', '9', '10'): 'Exercício 5',  
    ('11', '12', '11'): 'Exercício 6',     
}

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas as origens
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def index():
    return HTMLResponse(open("templates/index.html").read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    recent_gestures = []
    current_gesture = None
    current_gesture_start_time = None
    current_exercise = None
    detection_active = True

    while True:
        try:
            # Recebe os dados do frontend
            data = await websocket.receive_text()

            # Verificar se é uma mensagem de controle (ex: limpar sequência)
            if data.startswith("{"):
                message = json.loads(data)
                if message.get("action") == "clear":
                    recent_gestures = []
                    current_exercise = None
                    print("Sequência de gestos limpa!")
                    await websocket.send_text(json.dumps({
                        "hand_state": "Nenhum gesto detectado",
                        "exercise": "Nenhum",
                        "recent_gestures": []
                    }))
                    continue

            # Processa o frame
            img_data = base64.b64decode(data.split(",")[1])
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                print("Erro: Frame não pôde ser decodificado.")
                continue

            # Extrai landmarks da mão
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        landmarks.append([landmark.x, landmark.y, landmark.z])
                landmarks = np.array(landmarks)

                # Normaliza os landmarks
                palm_landmark = landmarks[0]
                normalized_landmarks = landmarks - palm_landmark
                max_value = np.max(np.abs(normalized_landmarks))
                if max_value > 0:
                    normalized_landmarks /= max_value

                # Extrai features
                features = np.concatenate([normalized_landmarks.flatten(), 
                                           calculate_distances(normalized_landmarks), 
                                           calculate_angles(normalized_landmarks)])

                # Pré-processamento dos landmarks
                input_data = np.expand_dims(features, axis=0)  # Adicionar dimensão do batch

                # Predição
                prediction = model.predict(input_data)
                predicted_class = np.argmax(prediction)
                gesture_name = class_names[predicted_class]

                # Verifica se o gesto atual é o mesmo que o último detectado
                if current_gesture == gesture_name:
                    # Verifica se o gesto foi mantido por tempo suficiente
                    if time.time() - current_gesture_start_time >= 2:  # 2 segundos
                        # Adiciona o gesto à lista de gestos recentes apenas se for diferente do último gesto na lista
                        if not recent_gestures or gesture_name != recent_gestures[-1]:
                            recent_gestures.append(gesture_name)

                        # Mante apenas os cinco gestos mais recentes
                        if len(recent_gestures) > 5:
                            recent_gestures.pop(0)

                        # Reseta o gesto atual e o tempo de início
                        current_gesture = None
                        current_gesture_start_time = None

                        # Verifica se a sequência de gestos corresponde a um exercício
                        exercise_name = check_exercise(recent_gestures)
                        if exercise_name:
                            current_exercise = exercise_name  # Atualiza o exercício atual
                            recent_gestures = []  # Zera o dicionário de gestos recentes
                else:
                    # Se o gesto mudou, atualizar o gesto atual e o tempo de início
                    current_gesture = gesture_name
                    current_gesture_start_time = time.time()

                # Envia a resposta para o frontend (estado da mão, exercício e sequência de gestos)
                response = {
                    "hand_state": gesture_name,
                    "exercise": current_exercise if current_exercise else "Nenhum",
                    "recent_gestures": recent_gestures
                }
                await websocket.send_text(json.dumps(response))

            else:
                # Se nenhuma mão for detectada
                response = {
                    "hand_state": "Nenhum gesto detectado",
                    "exercise": "Nenhum",
                    "recent_gestures": recent_gestures
                }
                await websocket.send_text(json.dumps(response))

        except Exception as e:
            print(f"Erro ao processar o WebSocket: {e}")
            break

def calculate_distances(landmarks):
    distances = []
    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            dist = np.linalg.norm(landmarks[i] - landmarks[j])
            distances.append(dist)
    return np.array(distances)

def calculate_angles(landmarks):
    angles = []
    for i in range(1, len(landmarks) - 1):
        v1 = landmarks[i] - landmarks[i - 1]
        v2 = landmarks[i + 1] - landmarks[i]
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angles.append(angle)
    return np.array(angles)

def check_exercise(recent_gestures):
    gesture_sequence = tuple(recent_gestures)
    if gesture_sequence in exercises:
        return exercises[gesture_sequence]
    else:
        return None

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)