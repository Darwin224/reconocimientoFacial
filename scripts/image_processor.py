import cv2
import numpy as np
import os
from typing import Tuple, List
import matplotlib.pyplot as plt

class ImageProcessor:
    """Clase para procesar imágenes para la red neuronal"""
    
    def __init__(self, target_size: Tuple[int, int] = (600, 800)):
        self.target_size = target_size
        self.width, self.height = target_size
    
    def load_and_preprocess(self, image_path: str) -> np.ndarray:
        """Cargar y preprocesar una imagen"""
        # Verificar que el archivo existe
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"No se encontró la imagen: {image_path}")
        
        # Leer imagen
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        # Redimensionar a tamaño objetivo
        img_resized = cv2.resize(img, self.target_size)
        
        # Convertir a escala de grises
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # Normalizar píxeles (0-1)
        img_normalized = img_gray.astype(np.float32) / 255.0
                
        # Aplanar para la red neuronal
        img_flattened = img_normalized.flatten() #convierte una imagen 2D A 1D
        
        return img_flattened

    def create_dataset_from_folder(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Crear dataset desde una carpeta con subcarpetas por clase"""
        X = []  # Lista para guardar las imágenes procesadas
        y = [] # Lista para guardar las etiquetas (labels) en formato one-hot
        class_names = [] # Lista para guardar los nombres de las clases
        
        # Obtener nombres de clases (subdirectorios)
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                class_names.append(item)
        
        class_names.sort() #ordena alfabeticamente
        print(f"Clases encontradas: {class_names}")
        
        # Procesar cada clase
        for class_idx, class_name in enumerate(class_names):
            class_path = os.path.join(data_dir, class_name)
            print(f"Procesando clase '{class_name}'...") #Obtiene las rutas de las clases
            
            image_count = 0
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_path, img_file)
                    
                    try:
                        # Procesar imagen
                        img_processed = self.load_and_preprocess(img_path)
                        X.append(img_processed)
                        
                        # Crear etiqueta one-hot
                        label = np.zeros(len(class_names))
                        label[class_idx] = 1
                        y.append(label)
                        
                        image_count += 1
                        
                    except Exception as e:
                        print(f"Error procesando {img_path}: {e}")
            
            print(f"  - {image_count} imágenes procesadas")
        
        print(f"Dataset creado: {len(X)} imágenes, {len(class_names)} clases")
        return np.array(X), np.array(y), class_names
    
    
    
    
    
    
    
