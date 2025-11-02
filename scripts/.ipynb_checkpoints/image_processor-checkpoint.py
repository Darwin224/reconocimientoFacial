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
        
        # Aplicar filtros para mejorar características
        img_processed = self.apply_filters(img_normalized)
        
        # Aplanar para la red neuronal
        img_flattened = img_processed.flatten()
        
        return img_flattened
    
    def apply_filters(self, img: np.ndarray) -> np.ndarray:
        """Aplicar filtros para mejorar características de la imagen"""
        # Filtro Gaussiano para suavizar
        img_blur = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Detección de bordes con Sobel
        sobel_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Combinar imagen original con detección de bordes
        img_enhanced = 0.7 * img_blur + 0.3 * sobel_combined
        
        # Normalizar resultado
        img_enhanced = np.clip(img_enhanced, 0, 1)
        
        return img_enhanced
    
    def create_dataset_from_folder(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Crear dataset desde una carpeta con subcarpetas por clase"""
        X = []
        y = []
        class_names = []
        
        # Obtener nombres de clases (subdirectorios)
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                class_names.append(item)
        
        class_names.sort()
        print(f"Clases encontradas: {class_names}")
        
        # Procesar cada clase
        for class_idx, class_name in enumerate(class_names):
            class_path = os.path.join(data_dir, class_name)
            print(f"Procesando clase '{class_name}'...")
            
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
    
    def visualize_preprocessing(self, image_path: str):
        """Visualizar el proceso de preprocesamiento"""
        # Cargar imagen original
        img_original = cv2.imread(image_path)
        img_original_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        
        # Redimensionar
        img_resized = cv2.resize(img_original, self.target_size)
        img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Escala de grises
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # Normalizar
        img_normalized = img_gray.astype(np.float32) / 255.0
        
        # Aplicar filtros
        img_processed = self.apply_filters(img_normalized)
        
        # Crear visualización
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(img_original_rgb)
        axes[0, 0].set_title('Imagen Original')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(img_resized_rgb)
        axes[0, 1].set_title(f'Redimensionada ({self.target_size[0]}x{self.target_size[1]})')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(img_gray, cmap='gray')
        axes[0, 2].set_title('Escala de Grises')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(img_normalized, cmap='gray')
        axes[1, 0].set_title('Normalizada (0-1)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(img_processed, cmap='gray')
        axes[1, 1].set_title('Con Filtros Aplicados')
        axes[1, 1].axis('off')
        
        # Histograma de píxeles
        axes[1, 2].hist(img_processed.flatten(), bins=50, alpha=0.7)
        axes[1, 2].set_title('Distribución de Píxeles')
        axes[1, 2].set_xlabel('Valor de Píxel')
        axes[1, 2].set_ylabel('Frecuencia')
        
        plt.tight_layout()
        plt.show()
    
    def augment_image(self, img: np.ndarray) -> List[np.ndarray]:
        """Aplicar data augmentation a una imagen"""
        augmented_images = []
        
        # Imagen original
        augmented_images.append(img)
        
        # Rotaciones
        for angle in [90, 180, 270]:
            if angle == 90:
                rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated = cv2.rotate(img, cv2.ROTATE_180)
            else:  # 270
                rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            augmented_images.append(rotated)
        
        # Flip horizontal
        flipped_h = cv2.flip(img, 1)
        augmented_images.append(flipped_h)
        
        # Flip vertical
        flipped_v = cv2.flip(img, 0)
        augmented_images.append(flipped_v)
        
        # Cambios de brillo
        bright = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
        dark = cv2.convertScaleAbs(img, alpha=0.8, beta=-10)
        augmented_images.extend([bright, dark])
        
        return augmented_images

# Ejemplo de uso
if __name__ == "__main__":
    processor = ImageProcessor(target_size=(600, 800))
    
    print("=== Procesador de Imágenes para Red Neuronal ===")
    print(f"Tamaño objetivo: {processor.target_size}")
    print()
    
    # Crear datos de ejemplo si no hay imágenes reales
    print("Para usar este procesador:")
    print("1. Crea una carpeta 'dataset' con subcarpetas por cada clase")
    print("2. Coloca las imágenes en las subcarpetas correspondientes")
    print("3. Ejecuta el procesamiento")
    print()
    
    # Ejemplo de estructura:
    print("Estructura de carpetas esperada:")
    print("dataset/")
    print("  ├── clase1/")
    print("  │   ├── imagen1.jpg")
    print("  │   └── imagen2.jpg")
    print("  ├── clase2/")
    print("  │   ├── imagen3.jpg")
    print("  │   └── imagen4.jpg")
    print("  └── clase3/")
    print("      ├── imagen5.jpg")
    print("      └── imagen6.jpg")
