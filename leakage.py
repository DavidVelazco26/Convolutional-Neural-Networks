import numpy as np
import matplotlib.pyplot as plt

# Parámetros del modelo
a = 4.0  # Coeficiente de fuga (leakage)
tau = 60  # Constante de tiempo

# Función para simular el leaky integrator
def leaky_integrator(input_signal):
    output = []
    x = 0.0  # Valor inicial del integrador
    for t in range(len(input_signal)):
        x = (1 - a) * x + input_signal[t]
        output.append(x)
    return output

# Generar una señal de entrada (por ejemplo, una señal escalonada)
t = np.arange(0, 50, 1)
input_signal = np.zeros(len(t))
input_signal[20:] = 1.0  # Señal escalonada

# Simular el leaky integrator con la señal de entrada
output_signal = leaky_integrator(input_signal)

# Graficar la señal de entrada y la salida del leaky integrator
plt.figure(figsize=(10, 6))
plt.plot(t, input_signal, label="Señal de Entrada")
plt.plot(t, output_signal, label="Salida del Integrador")
plt.title("Leaky Integrator")
plt.xlabel("Tiempo")
plt.ylabel("Valor")
plt.legend()
plt.show()
