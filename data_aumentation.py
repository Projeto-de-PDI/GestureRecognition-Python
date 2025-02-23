from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img
import os
import numpy as np
from PIL import Image
import random

# Defina o diretório de entrada corretamente
input_dir = "C:\\Users\\jmarq\\Desktop\\pdifinal\\GestureRecognition-Python\\data\\DATASET\\12"

# Defina os diretórios de saída para treino e validação
output_train_dir = "C:\\Users\\jmarq\\Desktop\\pdifinal\\GestureRecognition-Python\\data\\train\\12"
output_validation_dir = "C:\\Users\\jmarq\\Desktop\\pdifinal\\GestureRecognition-Python\\data\\val\\12"

# Cria os diretórios de saída se não existirem
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_validation_dir, exist_ok=True)

# Função para aplicar rotações específicas com fundo branco
def apply_rotation(image, angle):
    image = np.uint8(image)
    pil_image = Image.fromarray(image)
    rotated_img = pil_image.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    return rotated_img

# Função para inverter a imagem horizontalmente
def flip_image(image):
    return Image.fromarray(np.uint8(image)).transpose(Image.FLIP_LEFT_RIGHT)

# Lista de ângulos para rotação (de 0° a 360° em intervalos de 10°)
rotation_angles = list(range(0, 360, 10))

# Definição das transformações (exceto rotação, que será tratada separadamente)
augmentations = {
    "horizontal_flip": ImageDataGenerator(horizontal_flip=True),
    "vertical_flip": ImageDataGenerator(vertical_flip=True),
    "brightness": ImageDataGenerator(brightness_range=[0.2, 0.8]),  # Reduz a luminosidade
    "noise": ImageDataGenerator(preprocessing_function=lambda x: x + np.random.normal(0, 0.1, x.shape))  # Adiciona ruído
}

num_images_per_original = 30  # Número de imagens geradas por transformação (exceto rotação)

# Lista para armazenar todas as imagens geradas
all_images = []

for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(input_dir, filename)
        img = load_img(img_path)
        img_array = img_to_array(img)
        img_array = img_array.reshape((1,) + img_array.shape)

        print(f"Processando: {filename}")

        # Aplica rotações em múltiplos ângulos
        for angle in rotation_angles:
            rotated_img = apply_rotation(img_array[0], angle)
            # Salva a imagem rotacionada
            new_image_path = os.path.join(output_train_dir, f"{filename.split('.')[0]}_rotation_{angle}.jpg")
            save_img(new_image_path, img_to_array(rotated_img))
            all_images.append(new_image_path)
            print(f"Imagem salva: {new_image_path}")

            # Inverte a imagem rotacionada e salva
            flipped_img = flip_image(rotated_img)
            flipped_image_path = os.path.join(output_train_dir, f"{filename.split('.')[0]}_rotation_{angle}_flipped.jpg")
            save_img(flipped_image_path, img_to_array(flipped_img))
            all_images.append(flipped_image_path)
            print(f"Imagem invertida salva: {flipped_image_path}")

        # Aplica outras transformações (flip, brightness, noise)
        for aug_name, datagen in augmentations.items():
            i = 0
            for batch in datagen.flow(img_array, batch_size=1):
                # Salva a imagem transformada
                new_image_path = os.path.join(output_train_dir, f"{filename.split('.')[0]}_{aug_name}_{i}.jpg")
                save_img(new_image_path, batch[0])
                all_images.append(new_image_path)
                print(f"Imagem salva: {new_image_path}")

                # Inverte a imagem transformada e salva
                flipped_img = flip_image(batch[0])
                flipped_image_path = os.path.join(output_train_dir, f"{filename.split('.')[0]}_{aug_name}_{i}_flipped.jpg")
                save_img(flipped_image_path, img_to_array(flipped_img))
                all_images.append(flipped_image_path)
                print(f"Imagem invertida salva: {flipped_image_path}")

                i += 1
                if i >= num_images_per_original:
                    break

print("Concluído a geração de imagens")

# Separa as imagens em treino e validação
random.shuffle(all_images)  # Embaralha as imagens
split_index = int(0.75 * len(all_images))  # 75% para treino, 25% para validação

train_images = all_images[:split_index]
validation_images = all_images[split_index:]

# Move as imagens para as pastas de treino e validação
for image_path in train_images:
    os.rename(image_path, os.path.join(output_train_dir, os.path.basename(image_path)))

for image_path in validation_images:
    os.rename(image_path, os.path.join(output_validation_dir, os.path.basename(image_path)))

print("Imagens separadas em treino e validação")