import numpy as np
import matplotlib.pyplot as plt

# Función de activación ReLU
def relu(x):
    return np.maximum(0, x)

# Valores de entrada
x = np.linspace(-5, 5, 100)

# Diferentes valores de peso y bias
pesos = [-2.0, -1.0, 0.5, 1.0, 2.0]
biases = [-3.0, -1.0, 0.0, 1.0, 3.0]

# Crear una gráfica para mostrar la salida de la neurona con diferentes pesos y bias
plt.figure(figsize=(10, 6))

for peso, bias in zip(pesos, biases):
    y = relu(peso * x + bias)
    plt.plot(x, y, label=f"Peso = {peso}, Bias = {bias}")

plt.title("Salida de Neurona con Diferentes Pesos y Bias (ReLU)")
plt.xlabel("Entrada")
plt.ylabel("Salida")
plt.legend()


# Mostrar la gráfica
plt.show()
