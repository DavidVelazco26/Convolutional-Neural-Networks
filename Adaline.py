import numpy as np
import pandas as pd

class Adaline:
    def __init__(self, learning_rate=0.01, epochs=50):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        self.weights = np.zeros(1 + X.shape[1])
        self.cost = []

        for _ in range(self.epochs):
            output = self.net_input(X)
            errors = y - output
            self.weights[1:] += self.learning_rate * X.T.dot(errors)
            self.weights[0] += self.learning_rate * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# Cargar el conjunto de datos del iris desde un archivo CSV
data = pd.read_csv('iris.csv')

# Seleccionar dos clases para la clasificación binaria (por ejemplo, setosa y versicolor)
X = data.iloc[0:100, [0, 2]].values
y = data.iloc[0:100, 4].values
y = np.where(y == 'setosa', -1, 1)  # Codificar las clases como -1 y 1

# Crear una instancia del modelo Adaline y entrenarlo
model = Adaline(learning_rate=0.01, epochs=50)
model.fit(X, y)

# Realizar una predicción
new_x = np.array([5.1, 1.4])
prediction = model.predict(new_x)

if prediction == -1:
    class_label = 'setosa'
else:
    class_label = 'versicolor'

print(f'Predicción para las características {new_x}: Clase predicha: {class_label}')
