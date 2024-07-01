import numpy as np
import matplotlib.pyplot as plt
## controlan la tasa de fuga
a = 0.1 # Coeficiente de fuga (leakage)
b = 1.0   # Escala de entrada
c = 0.7   # Ganancia de retroalimentación (inhibición)
tau = 20  # Retraso en el tiempo

# Función para simular el DLI
def delayed_leak_integrator(input_signal):
    y = np.zeros(len(input_signal))
    for t in range(len(input_signal)):
        if t < tau:
            delayed_input = 0.0
        else:
            delayed_input = input_signal[t - tau]
        y[t] = -a * y[t - 1] + b * input_signal[t] - c * delayed_input
    return y

# Generar una señal de entrada suave
t = np.arange(0, 150, 1)
input_signal = np.sin(0.1 * t) + 0.5 * np.sin(0.5 * t)
#input_signal = max(0,t)
# Simular el DLI con la señal de entrada
output_signal = delayed_leak_integrator(input_signal)

# Graficar la señal de entrada y la salida en la misma gráfica
plt.figure(figsize=(10, 6))
plt.plot(t, input_signal, label="Señal de Entrada")
plt.plot(t, output_signal, color='red', label="Salida del DLI")
plt.title("Señal de Entrada y Salida del DLI")
plt.xlabel("Tiempo")
plt.ylabel("Valor")
plt.legend()
plt.show()
