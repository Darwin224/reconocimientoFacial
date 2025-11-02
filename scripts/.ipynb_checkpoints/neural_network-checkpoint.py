import numpy as np
import json

class RedNeuronal:
    def __init__(self, architecture):
        self.architecture = architecture
        self.weights = []
        self.biases = []
        self.training_history = []
        
        # Validación: la primera capa debe ser de tipo 'input'
        if architecture[0][1] != 'input':
            raise ValueError("La primera capa debe tener tipo 'input'.")
        
        # Inicialización de pesos y sesgos
        for i in range(len(architecture) - 1):
            input_dim = architecture[i][0]
            output_dim = architecture[i + 1][0]
            w = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)  # He para ReLU
            b = np.zeros((1, output_dim))
            self.weights.append(w)
            self.biases.append(b)

    def _activation(self, x, func):
        if func == 'relu':
            return np.maximum(0, x)
        elif func == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif func == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        else:
            raise ValueError(f"Función de activación desconocida: {func}")

    def _activation_derivative(self, x, func):
        if func == 'relu':
            return (x > 0).astype(float)
        elif func == 'sigmoid':
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        else:
            raise ValueError(f"No se puede derivar la función: {func}")

    def _forward(self, X):
        activations = [X]
        zs = []
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            zs.append(z)
            activation_func = self.architecture[i + 1][1]  # Saltamos la capa de entrada
            a = self._activation(z, activation_func)
            activations.append(a)
        return activations, zs

    def _backward(self, activations, zs, y_true):
        grads_w = []
        grads_b = []
        m = y_true.shape[0]
        
        # Última capa (softmax + cross-entropy)
        delta = activations[-1] - y_true
        dw = np.dot(activations[-2].T, delta) / m
        db = np.sum(delta, axis=0, keepdims=True) / m
        grads_w.insert(0, dw)
        grads_b.insert(0, db)
        
        # Capas ocultas
        for i in reversed(range(len(self.weights) - 1)):
            activation_func = self.architecture[i + 1][1]
            delta = np.dot(delta, self.weights[i + 1].T) * self._activation_derivative(zs[i], activation_func)
            dw = np.dot(activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            grads_w.insert(0, dw)
            grads_b.insert(0, db)
        
        return grads_w, grads_b

    def train(self, X, y, epochs=200, learning_rate=0.001, batch_size=8, verbose=True):
        m = X.shape[0]
        for epoch in range(1, epochs + 1):
            indices = np.arange(m)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            epoch_accuracy = 0
            
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                activations, zs = self._forward(X_batch)
                loss = self._cross_entropy(activations[-1], y_batch)
                grads_w, grads_b = self._backward(activations, zs, y_batch)
                
                for j in range(len(self.weights)):
                    self.weights[j] -= learning_rate * grads_w[j]
                    self.biases[j] -= learning_rate * grads_b[j]
                
                epoch_loss += loss * X_batch.shape[0]
                predictions = np.argmax(activations[-1], axis=1)
                true_labels = np.argmax(y_batch, axis=1)
                epoch_accuracy += np.sum(predictions == true_labels)
            
            epoch_loss /= m
            epoch_accuracy /= m
            self.training_history.append((epoch_loss, epoch_accuracy))
            
            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.4f}")

    def _cross_entropy(self, pred, label):
        epsilon = 1e-12
        pred = np.clip(pred, epsilon, 1. - epsilon)
        return -np.mean(np.sum(label * np.log(pred), axis=1))

    def predict(self, X):
        activations, _ = self._forward(X)
        return activations[-1]

    def evaluate(self, X, y):
        predictions = self.predict(X)
        loss = self._cross_entropy(predictions, y)
        pred_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(pred_labels == true_labels)
        return loss, accuracy
    def load_model(self, filename):
        """🔥 MÉTODO AÑADIDO - Cargar un modelo previamente guardado"""
        try:
            with open(filename, 'r') as f:
                model_data = json.load(f)
            
            # Cargar arquitectura
            self.architecture = model_data['architecture']
            
            # Cargar pesos y biases
            self.weights = [np.array(w) for w in model_data['weights']]
            self.biases = [np.array(b) for b in model_data['biases']]
            
            # Inicializar historial vacío
            self.training_history = []
            
            print(f"✅ Modelo cargado desde: {filename}")
            print(f"📊 Arquitectura: {len(self.architecture)} capas")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ No se encontró el archivo: {filename}")
        except KeyError as e:
            raise KeyError(f"❌ Formato de archivo inválido. Falta clave: {e}")
        except Exception as e:
            raise Exception(f"❌ Error cargando modelo: {e}")

    def save_model(self, filename):
        """Guardar el modelo entrenado"""
        model_data = {
            'architecture': self.architecture,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases]
        }
        with open(filename, 'w') as f:
            json.dump(model_data, f, indent=2)
        print(f"✅ Modelo guardado en: {filename}")

    

