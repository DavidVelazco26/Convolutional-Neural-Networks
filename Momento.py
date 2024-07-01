import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

# Cargar el conjunto de datos de diabetes
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target.reshape(-1, 1)

# Normalizar los datos
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Agregar una columna de unos para el término de sesgo (bias)
X_bias = np.c_[np.ones((X.shape[0], 1)), X]

def gradient_descent_with_momentum(X, y, learning_rate=0.01, beta=0.9, epochs=100):
    m, n = X.shape  # m: número de ejemplos, n: número de características
    theta = np.zeros((n, 1))  # Inicialización de parámetros
    v = np.zeros_like(theta)  # Inicialización del término de momento

    for epoch in range(epochs):
        # Calcular el gradiente
        gradients = compute_gradient(X, y, theta)

        # Actualizar el término de momento
        v = beta * v + (1 - beta) * gradients

        # Actualizar los parámetros
        theta = theta - learning_rate * v

        # Calcular la función de pérdida (opcional, solo para seguimiento)
        loss = compute_loss(X, y, theta)

        # Imprimir la pérdida en cada iteración (opcional)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss}')

    return theta

def compute_gradient(X, y, theta):
    m = X.shape[0]
    h = X @ theta  # Producto punto de X y theta
    error = h - y
    gradient = (1 / m) * (X.T @ error)  # Gradiente con respecto a theta
    return gradient

def compute_loss(X, y, theta):
    m = X.shape[0]
    h = X @ theta  # Producto punto de X y theta
    error = h - y
    loss = (1 / (2 * m)) * np.sum(error**2)  # Función de pérdida (MSE)
    return loss

# Llamar a la función de descenso de gradiente con momento
theta_optimized = gradient_descent_with_momentum(X_bias, y)
print("Parámetros optimizados:", theta_optimized)
