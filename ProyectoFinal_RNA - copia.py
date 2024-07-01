import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Función de fitness (objetivo a maximizar)
def fitness_function(params):
    # Creamos o actualizamos el modelo con los nuevos parámetros
    model = QNetwork(input_size, output_size, params)
    
    # Entrenamos y evaluamos el modelo
    total_reward = train_and_evaluate(model)
    
    # Devolvemos el total_reward como medida de aptitud (mayor es mejor)
    return total_reward

# Modelo de la red neuronal
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, params):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

        # Aplicamos los parámetros dados a las capas lineales
        self.apply_params(params)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def apply_params(self, params):
        # Aplicamos los parámetros a las capas lineales
        self.fc1.weight.data = torch.tensor(params['fc1.weight'], dtype=torch.float32)
        self.fc1.bias.data = torch.tensor(params['fc1.bias'], dtype=torch.float32)
        self.fc2.weight.data = torch.tensor(params['fc2.weight'], dtype=torch.float32)
        self.fc2.bias.data = torch.tensor(params['fc2.bias'], dtype=torch.float32)

# Entrenamiento y evaluación del modelo
def train_and_evaluate(model):
    # Implementa aquí tu lógica de entrenamiento y evaluación
    # Retorna la recompensa total como medida de aptitud

# Configuración
input_size = 4  # Cambia esto según tus necesidades
output_size = 2  # Cambia esto según tus necesidades
num_params = sum(p.numel() for p in QNetwork(input_size, output_size, {}).parameters())  # Número total de parámetros

# Parámetros del algoritmo genético
population_size = 10
mutation_rate = 0.1
num_generations = 100

# Inicialización aleatoria de la población
population = [{'fc1.weight': np.random.randn(64, input_size),
               'fc1.bias': np.random.randn(64),
               'fc2.weight': np.random.randn(output_size, 64),
               'fc2.bias': np.random.randn(output_size)}
              for _ in range(population_size)]

# Algoritmo genético de estado estacionario
for generation in range(num_generations):
    # Evaluar la aptitud de cada individuo en la población
    fitness_scores = [fitness_function(params) for params in population]

    # Seleccionar dos padres basados en torneo
    parents_indices = np.random.choice(population_size, size=(2, population_size), replace=True)
    parents = [population[i] for i in np.argmax(np.array(fitness_scores)[parents_indices], axis=1)]

    # Cruzar los padres para crear dos hijos
    crossover_point = np.random.randint(1, num_params - 1)
    child1 = {key: np.concatenate((parents[0][key][:crossover_point], parents[1][key][crossover_point:]))
              for key in parents[0]}
    child2 = {key: np.concatenate((parents[1][key][:crossover_point], parents[0][key][crossover_point:]))
              for key in parents[1]}

    # Aplicar mutación a los hijos
    if np.random.rand() < mutation_rate:
        for child in [child1, child2]:
            for key in child:
                child[key] += np.random.randn(*child[key].shape) * 0.1

    # Reemplazar a los dos peores individuos en la población con los hijos
    worst_indices = np.argsort(fitness_scores)[:2]
    population[worst_indices[0]] = child1
    population[worst_indices[1]] = child2

    # Imprimir información sobre la generación
    print(f"Generation: {generation + 1}, Best Fitness: {max(fitness_scores)}")

# Obtener los mejores parámetros
best_params = population[np.argmax(fitness_scores)]
best_model = QNetwork(input_size, output_size, best_params)

# Realizar evaluación final del mejor modelo
best_fitness = fitness_function(best_params)
print(f"\nBest Model Fitness: {best_fitness}")
