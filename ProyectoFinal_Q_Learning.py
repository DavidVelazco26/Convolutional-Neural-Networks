import gym
import numpy as np

# Configuración del entorno
env = gym.make('MountainCar-v0')

# Parámetros de Q-learning
learning_rate = 0.1
discount_factor = 0.99
exploration_prob = 1.0  # Probabilidad de exploración inicial
exploration_decay = 0.995  # Factor de decaimiento de la probabilidad de exploración
min_exploration_prob = 0.1  # Probabilidad mínima de exploración

# Discretización del espacio de observaciones
num_discrete_states = [20, 20]
state_bins = [np.linspace(env.observation_space.low[i], env.observation_space.high[i], num_discrete_states[i] + 1)[1:-1] for i in range(env.observation_space.shape[0])]
action_space_size = env.action_space.n

# Inicialización de la tabla Q
Q_table = np.random.uniform(low=-1, high=1, size=(num_discrete_states + [action_space_size]))

def discretize_state(state):
    discrete_state = [np.digitize(state[i], state_bins[i]) for i in range(env.observation_space.shape[0])]
    return tuple(discrete_state)

def choose_action(discrete_state):
    global exploration_prob
    if np.random.rand() < exploration_prob:
        return env.action_space.sample()  # Exploración aleatoria
    else:
        return np.argmax(Q_table[discrete_state])

def update_q_table(discrete_state, action, reward, next_discrete_state):
    best_next_action = np.argmax(Q_table[next_discrete_state])
    current_q = Q_table[discrete_state + (action,)]
    new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * Q_table[next_discrete_state + (best_next_action,)])
    Q_table[discrete_state + (action,)] = new_q

# Entrenamiento del agente
num_episodes = 100

for episode in range(num_episodes):
    state = env.reset()
    discrete_state = discretize_state(state)

    done = False
    total_reward = 0

    while not done:
        # env.render()  # Descomenta esta línea para visualizar la simulación

        action = choose_action(discrete_state)
        next_state, reward, done, _ = env.step(action)
        next_discrete_state = discretize_state(next_state)

        update_q_table(discrete_state, action, reward, next_discrete_state)

        discrete_state = next_discrete_state
        total_reward += reward

    # Imprimir información sobre el episodio
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Exploration Probability: {exploration_prob}")

    # Decaimiento de la probabilidad de exploración
    exploration_prob = max(min_exploration_prob, exploration_prob * exploration_decay)

env.close()
