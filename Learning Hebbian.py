import numpy as np

# Definir la arquitectura neuronal
num_entradas = 3
num_salidas = 2
tasa_aprendizaje = 0.1

# Inicializar la matriz de pesos sinápticos
pesos_sinapticos = np.zeros((num_entradas, num_salidas))

# Definir los patrones de entrada
patrones_entrada = np.array([[1, 1, 0],
                              [0, 1, 1],
                              [1, 1, 1]])

# Definir los patrones de salida deseada
patrones_salida_deseada = np.array([[1, 1],
                                    [1, 1],
                                    [1, 1]])

# Entrenamiento usando aprendizaje Hebbiano
for i in range(len(patrones_entrada)):
    entrada = patrones_entrada[i]
    salida_deseada = patrones_salida_deseada[i]
    
    # Calcular la salida de la red
    salida_real = np.dot(entrada, pesos_sinapticos)
    
    # Actualizar los pesos sinápticos utilizando la regla de Hebb
    delta_pesos = tasa_aprendizaje * np.outer(entrada, salida_deseada - salida_real)
    pesos_sinapticos += delta_pesos

# Probar el controlador entrenado
def probar_controlador(entrada):
    return np.dot(entrada, pesos_sinapticos)

# Ejemplos de pruebas
entrada_prueba = np.array([1, 1, 0])
salida_prueba = probar_controlador(entrada_prueba)
print("Entrada de prueba:", entrada_prueba)
print("Salida esperada:", patrones_salida_deseada[0])
print("Salida real:", salida_prueba)

entrada_prueba = np.array([0, 1, 1])
salida_prueba = probar_controlador(entrada_prueba)
print("Entrada de prueba:", entrada_prueba)
print("Salida esperada:", patrones_salida_deseada[1])
print("Salida real:", salida_prueba)

entrada_prueba = np.array([1, 1, 1])
salida_prueba = probar_controlador(entrada_prueba)
print("Entrada de prueba:", entrada_prueba)
print("Salida esperada:", patrones_salida_deseada[2])
print("Salida real:", salida_prueba)
