import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np

# Configuración del entorno
env = gym.make('MountainCar-v0')

# Definición del modelo de la red neuronal
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Configuración de la red neuronal
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
learning_rate = 0.001

# Construcción del modelo
model = QNetwork(input_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Función para preprocesar el estado
def preprocess_state(state):
    return torch.tensor(state, dtype=torch.float32).view(1, -1)

# Entrenamiento de la red neuronal
num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    state = preprocess_state(state)

    done = False
    total_reward = 0

    while not done:
        # env.render()  # Descomenta esta línea para visualizar la simulación

        # Seleccionar una acción utilizando la red neuronal
        action_values = model(state)
        action = torch.argmax(action_values).item()

        # Tomar la acción y observar el siguiente estado y recompensa
        next_state, reward, done, _ = env.step(action)
        next_state = preprocess_state(next_state)

        # Calcular la recompensa objetivo (target)
        with torch.no_grad():
            target = reward + 0.99 * torch.max(model(next_state))

        # Calcular la pérdida y realizar la retropropagación
        loss = nn.MSELoss()(model(state)[0, action], target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        total_reward += reward

    # Imprimir información sobre el episodio
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

env.close()
