import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time
import keyboard  # Biblioteca para detectar eventos de teclado

# Inicializar o módulo de detecção de mãos do MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Carregando o modelo treinado
model = load_model('models/hand_gesture_landmarks_model.h5')  

# Mapeamento das classes
class_names = {0: '0', 1: '1', 2: '2', 3: '3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'10', 11:'11', 12:'12'}

# Dicionário de exercícios (sequência de gestos -> nome do exercício)
exercises = {
    ('0', '1', '0'): 'Exercício 1',  
    ('0', '2', '3', '4'): 'Exercício 2',
    ('0', '5', '0'): 'Exercício 3',           
    ('0', '6', '0'): 'Exercício 4',  
    ('0', '7', '8', '9', '10'): 'Exercício 5',  
    ('11', '12', '11'): 'Exercício 6',     
}


# Variáveis para armazenar os gestos recentes e controlar o tempo
recent_gestures = []
last_gesture_time = 0
gesture_hold_time = 2  # Tempo que o gesto deve ser mantido para ser considerado válido (em segundos)
current_gesture = None
current_gesture_start_time = None
current_exercise = None  # Variável para armazenar o exercício atual detectado
detection_active = False  # Variável para controlar se a detecção está ativa

# Função para Normalizando os landmarks em relação à palma da mão
def normalize_landmarks(landmarks):
    # Usando o landmark da palma (índice 0) como referência
    palm_landmark = landmarks[0]
    
    # Normalizando os landmarks em relação à palma
    normalized_landmarks = landmarks - palm_landmark
    
    # Ajustando a escala para que os landmarks estejam em uma faixa consistente
    max_value = np.max(np.abs(normalized_landmarks))
    if max_value > 0:
        normalized_landmarks /= max_value
    
    return normalized_landmarks

# Função para calcular distâncias entre landmarks
def calculate_distances(landmarks):
    distances = []
    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            dist = np.linalg.norm(landmarks[i] - landmarks[j])
            distances.append(dist)
    return np.array(distances)

# Função para calcular ângulos entre landmarks
def calculate_angles(landmarks):
    angles = []
    for i in range(1, len(landmarks) - 1):
        v1 = landmarks[i] - landmarks[i - 1]
        v2 = landmarks[i + 1] - landmarks[i]
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angles.append(angle)
    return np.array(angles)

# Função para Extraindo features (landmarks, distâncias e ângulos)
def extract_features(landmarks):
    distances = calculate_distances(landmarks)
    angles = calculate_angles(landmarks)
    return np.concatenate([landmarks.flatten(), distances, angles])

# Função para Extraindo landmarks da mão
def extract_landmarks(frame):
    # Converter o frame para RGB (MediaPipe requer RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Processar o frame para detectar landmarks
    results = hands.process(frame_rgb)
    
    # Verificar se landmarks foram detectados
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            # Extraindo coordenadas (x, y, z) de cada landmark
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
        landmarks = np.array(landmarks)
        
        # Normalizando os landmarks
        normalized_landmarks = normalize_landmarks(landmarks)
        
        # Extraindo features
        features = extract_features(normalized_landmarks)
        return features
    else:
        return None

# Função para verificar se a sequência de gestos corresponde a um exercício
def check_exercise(recent_gestures):
    # Converte a lista de gestos recentes para uma tupla (para ser usada como chave no dicionário)
    gesture_sequence = tuple(recent_gestures)
    
    # Verifica se a sequência está no dicionário de exercícios
    if gesture_sequence in exercises:
        return exercises[gesture_sequence]  # Retorna o nome do exercício
    else:
        return None  # Nenhum exercício correspondente

# Captura de vídeo
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Exibir mensagem de instrução na tela
    if not detection_active:
        cv2.putText(frame, 'Pressione "s" para iniciar a detecção', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'Pressione "p" para pausar a detecção', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Verificar se a detecção está ativa
    if detection_active:
        # Extraindo landmarks da mão
        features = extract_landmarks(frame)

        if features is not None:
            # Pré-processamento dos landmarks
            input_data = np.expand_dims(features, axis=0)  # Adicionar dimensão do batch

            # Predição
            prediction = model.predict(input_data)
            predicted_class = np.argmax(prediction)
            gesture_name = class_names[predicted_class]

            # Verificar se o gesto atual é o mesmo que o último detectado
            if current_gesture == gesture_name:
                # Verificar se o gesto foi mantido por tempo suficiente
                if time.time() - current_gesture_start_time >= gesture_hold_time:
                    # Adicionar o gesto à lista de gestos recentes apenas se for diferente do último gesto na lista
                    if not recent_gestures or gesture_name != recent_gestures[-1]:
                        recent_gestures.append(gesture_name)
                    
                    # Manter apenas os cinco gestos mais recentes
                    if len(recent_gestures) > 5:
                        recent_gestures.pop(0)
                    
                    # Resetar o gesto atual e o tempo de início
                    current_gesture = None
                    current_gesture_start_time = None

                    # Verificar se a sequência de gestos corresponde a um exercício
                    exercise_name = check_exercise(recent_gestures)
                    if exercise_name:
                        current_exercise = exercise_name  # Atualiza o exercício atual
                        recent_gestures = []  # Zera o dicionário de gestos recentes
                        print(f"Exercício detectado: {exercise_name}")  # Exibe no console
            else:
                # Se o gesto mudou, atualizar o gesto atual e o tempo de início
                current_gesture = gesture_name
                current_gesture_start_time = time.time()

            # Exibir o gesto atual e os gestos recentes na tela
            cv2.putText(frame, f'Gesto: {gesture_name}', (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Ultimos Gestos: {", ".join(recent_gestures)}', (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Exibir o exercício atual na tela
            if current_exercise:
                cv2.putText(frame, f'Exercício Atual: {current_exercise}', (10, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            # Se nenhuma mão for detectada
            cv2.putText(frame, 'Nenhum gesto detectado', (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            current_exercise = None  # Limpa o exercício atual se nenhuma mão for detectada

    # Exibir o frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Verificar se a tecla 's' foi digitada para iniciar a detecção
    if keyboard.is_pressed('s'):
        detection_active = True  # Ativa a detecção de gestos
        print("Detecção de gestos iniciada!")  # Exibe no console

    # Verificar se a tecla 'p' foi digitada para pausar a detecção
    if keyboard.is_pressed('p'):
        detection_active = False  # Pausa a detecção de gestos
        print("Detecção de gestos pausada!")  # Exibe no console

    # Verificar se a tecla 'r' foi digitada para resetar o dicionário de gestos recentes
    if keyboard.is_pressed('r'):
        recent_gestures = []  # Zera a lista de gestos recentes
        current_exercise = None  # Limpa o exercício atual
        print("Dicionário de gestos recentes zerado!")  # Exibe no console

    # Parar ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()