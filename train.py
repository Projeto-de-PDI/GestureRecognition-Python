import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight

# Inicializa o módulo de detecção de mãos do MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Função para extrair landmarks da mão
def extract_landmarks(image):
    # Converter a imagem para RGB (MediaPipe requer RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Processa a imagem para detectar landmarks
    results = hands.process(image_rgb)
    
    # Verifica se landmarks foram detectados
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            # Extrai coordenadas (x, y, z) de cada landmark
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)
    else:
        return None

# Função para normalizar os landmarks em relação à palma da mão
def normalize_landmarks(landmarks):
    # Usar o landmark da palma (índice 0) como referência
    palm_landmark = landmarks[0]
    
    # Normaliza os landmarks em relação à palma
    normalized_landmarks = landmarks - palm_landmark
    
    # Ajusta a escala para que os landmarks estejam em uma faixa consistente
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

# Função para extrair features (landmarks, distâncias e ângulos)
def extract_features(landmarks):
    distances = calculate_distances(landmarks)
    angles = calculate_angles(landmarks)
    return np.concatenate([landmarks.flatten(), distances, angles])

# Função para aumentar dados (rotação, translação, escala)
def augment_landmarks(landmarks):
    angle = np.random.uniform(-10, 10)  # Rotação
    scale = np.random.uniform(0.9, 1.1)  # Escala
    translation = np.random.uniform(-0.1, 0.1, size=(3,))  # Translação
    
    # Aplica transformações
    rotation_matrix = np.array([
        [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
        [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0],
        [0, 0, 1]
    ])
    landmarks = np.dot(landmarks, rotation_matrix) * scale + translation
    return landmarks

# Função para carregar imagens e extrair features
def load_images_and_features(data_dir, augment=False):
    features_list = []
    labels = []
    classes = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,  '7': 7,'8': 8,  '9': 9,'10': 10, '11': 11,'12': 12 }

    for class_name, class_id in classes.items():
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            
            # Extrai landmarks da imagem
            landmarks = extract_landmarks(img)
            if landmarks is not None:
                # Normaliza os landmarks
                normalized_landmarks = normalize_landmarks(landmarks)
                
                # Extrai features
                features = extract_features(normalized_landmarks)
                features_list.append(features)
                labels.append(class_id)
                
                # Aumenta dados (opcional)
                if augment:
                    augmented_landmarks = augment_landmarks(normalized_landmarks)
                    augmented_features = extract_features(augmented_landmarks)
                    features_list.append(augmented_features)
                    labels.append(class_id)
            else:
                print(f"Landmarks não detectados em: {img_path}")

    return np.array(features_list), np.array(labels)

# Carrega dados
train_features, train_labels = load_images_and_features('data/train', augment=True)
val_features, val_labels = load_images_and_features('data/val')

# Verifica se há dados suficientes
if len(train_features) == 0 or len(val_features) == 0:
    raise ValueError("Nenhum landmark foi detectado. Verifique as imagens e a detecção de mãos.")

# Balanceamento de classes
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = dict(enumerate(class_weights))

# Defini o modelo MLP
model = models.Sequential([
    layers.Input(shape=(21 * 3 + 210 + 19,)),  # 21 landmarks + 210 distâncias + 19 ângulos
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(13, activation='softmax')  # 13 classes
])

# Compila o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model_landmarks.h5', monitor='val_accuracy', save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# Treinar o modelo
history = model.fit(train_features, train_labels, batch_size=32, epochs=30,
                    validation_data=(val_features, val_labels),
                    class_weight=class_weights,
                    callbacks=[early_stopping, model_checkpoint, reduce_lr])

# Salva o modelo final
model.save('models/hand_gesture_landmarks_model.h5')

# Plota curvas de treinamento
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Curva de Perda')
plt.show()

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Curva de Acurácia')
plt.show()