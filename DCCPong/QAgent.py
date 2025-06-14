
#################################################
#                                               #
#            Actividad 2 de código              #
#                                               #
#  Implementación de un agente Q-Learning para  #
#                el juego Pong                  #
#                                               #
#     Tiempo con visualizacion: 5 minutos       #
#                                               #
#################################################



import numpy as np
import os
from PongAI import PongAI


os.environ['SDL_AUDIODRIVER'] = 'dummy'  # Desactiva el audio de pygame para evitar errores


# Para completar la Q-Table en el código de TCouso, puede ser revisado el siguiente enlace:
# https://github.com/tcouso/tareas-iic2613/tree/d8d31a3742c2d9b2bdb29331ba9945efff355197/tarea-4-2022-1-tcouso/02%20-%20Aprendizaje%20Reforzado

VEL_BINS = 11
PROXIMITY_BINS = 6
HITPOINT_BINS = 4  
HITPOINT_COMBINED_BINS = HITPOINT_BINS * HITPOINT_BINS  
BOUNCE_BINS = 3


# Parámetros de la Q-Table

NUM_EPISODES = 1000

LR = .05
DISCOUNT_RATE = .9
MAX_EXPLORATION_RATE = 1
MIN_EXPLORATION_RATE = .0001
EXPLORATION_DECAY_RATE = .0005



N_STATES = VEL_BINS * PROXIMITY_BINS * HITPOINT_COMBINED_BINS * BOUNCE_BINS  

N_ACTIONS = 3  # 0: No hacer nada, 1: Mover hacia arriba, 2: Mover hacia abajo

# Si deseas o no tener elementos gráficos del juego (más lento si se muestran)
VISUALIZATION = False


class Agent:
    # Esta clase posee al agente y define sus comportamientos.

    def __init__(self):
        # Creamos la q_table y la inicializamos en 0.
        # IMPLEMENTAR
        self.q_table =  np.zeros((N_STATES, N_ACTIONS))

        # Inicializamos los juegos realizados por el agente en 0.
        self.n_games = 0

        # Inicializamos el exploration rate.
        self.exploration_rate = MAX_EXPLORATION_RATE
    
    def state_to_index(self, state):
        velocity, proximity, up_state, down_state, bounces = state  # Sin paddle

        velocity = int((velocity + 5) / 10 * (VEL_BINS - 1))
        proximity = int((proximity / 5) * (PROXIMITY_BINS - 1))
        up_state = int((up_state / 3) * (HITPOINT_BINS - 1))
        down_state = int((down_state / 3) * (HITPOINT_BINS - 1))
        bounces = int((bounces / 2) * (BOUNCE_BINS - 1))

        velocity = max(0, min(velocity, VEL_BINS - 1))
        proximity = max(0, min(proximity, PROXIMITY_BINS - 1))
        up_state = max(0, min(up_state, HITPOINT_BINS - 1))
        down_state = max(0, min(down_state, HITPOINT_BINS - 1))
        bounces = max(0, min(bounces, BOUNCE_BINS - 1))

        # Combinamos up_state y down_state en un solo índice
        hitpoint_combined = up_state * HITPOINT_BINS + down_state

        index = velocity
        index = index * PROXIMITY_BINS + proximity
        index = index * HITPOINT_COMBINED_BINS + hitpoint_combined
        index = index * BOUNCE_BINS + bounces

        return index
    
    def update_q_table(self, move, reward, state, new_state):
        # Convertimos los estados a índices de la Q-Table
        state_index = self.state_to_index(state)
        new_state_index = self.state_to_index(new_state)

        # Valor anterior en la Q-Table para la acción tomada
        old_value = self.q_table[state_index, move]

        # Estimación del valor futuro (usando la mejor acción futura)
        future_reward = np.max(self.q_table[new_state_index, :])

        # Q-Learning: actualizamos el valor con la fórmula de Bellman
        new_value = (1 - LR) * old_value + LR * (reward + DISCOUNT_RATE * future_reward)
        self.q_table[state_index, move] = new_value
    

    def get_state(self, game):
        # Este método consulta al juego por el estado del agente y lo retorna como una tupla.
        state = []

        # Obtenemos la velocidad en y de la pelota
        velocity = int(round(game.ball.y_vel, 0))
        state.append(velocity)

        # Obtenemos el cuadrante de la pelota
        proximity = 5 - int(round(game.ball.x / game.MAX_X) * 5)
        state.append(proximity)

        # Revisamos la posición de la pelota respecto al extremo superior del agente  
        if game.ball.y < (game.right_paddle.y):
            if game.right_paddle.y - game.ball.y > game.right_paddle.height:
                up_state = 0
            else:
                up_state = 1
        else:
            if game.ball.y - game.right_paddle.y < game.right_paddle.height:
                up_state = 2
            else:
                up_state = 3
        state.append(up_state)

        # Revisamos la posición de la pelota respecto al extremo inferior del agente 
        if game.ball.y < (game.right_paddle.y + game.right_paddle.height):
            if game.right_paddle.y + game.right_paddle.height - game.ball.y > game.right_paddle.height:
                down_state = 0
            else:
                down_state = 1
        else:
            if game.ball.y - game.right_paddle.y - game.right_paddle.height < game.right_paddle.height:
                down_state = 2
            else:
                down_state = 3
        state.append(down_state)

        # Número de botes contra la pared que ha dado la pelota
        bounces = game.ball.bounces
        state.append(bounces)


        return tuple(state)

    def get_action(self, state):
        # Este método recibe una estado del agente y retorna un entero con el indice de la acción correspondiente.

        exploration_prob = self.exploration_rate
        state_idx = self.state_to_index(state)

        if np.random.rand() < exploration_prob:
            action = np.random.choice([0, 1, 2])
        else:
            action = np.argmax(self.q_table[state_idx])
        return action
    
    def update_exploration_rate(self):
        self.exploration_rate = MIN_EXPLORATION_RATE + (MAX_EXPLORATION_RATE - MIN_EXPLORATION_RATE) * np.exp(-EXPLORATION_DECAY_RATE*self.n_games)


def index_to_state(index):
    bounces = index % BOUNCE_BINS
    index //= BOUNCE_BINS

    hitpoint_combined = index % HITPOINT_COMBINED_BINS
    index //= HITPOINT_COMBINED_BINS

    proximity = index % PROXIMITY_BINS
    index //= PROXIMITY_BINS

    velocity = index % VEL_BINS

    # Separamos hitpoint_combined en up_state y down_state
    up_state = hitpoint_combined // HITPOINT_BINS
    down_state = hitpoint_combined % HITPOINT_BINS

    return velocity, proximity, up_state, down_state, bounces

def train():
    plot_scores = []
    plot_mean_scores = []
    mean_score = 0
    total_score = 0
    record = 0
    period_steps = 0
    period_score = 0

    training_log = []  # Guardará Partida, Puntaje promedio y Exploration Rate

    agent = Agent()
    game = PongAI(vis=VISUALIZATION)
    steps = 0

    while True:
        state = agent.get_state(game)
        move = agent.get_action(state)
        reward, done, score = game.play_step(move)
        new_state = agent.get_state(game)

        agent.update_q_table(move, reward, state, new_state)

        if agent.n_games % 500 == 0 and agent.n_games > 0:
            print(f"DEBUG (Episode {agent.n_games}): State={state}, Move={move}, Reward={reward}, Score={score}")

        print(f"Game {agent.n_games} | Score: {score} | Steps: {steps} | Exploration Rate: {agent.exploration_rate:.4f}")
        if done:
            agent.update_exploration_rate()
            game.reset()
            agent.n_games += 1

            if agent.n_games % 100 == 0:
                np.save("q_table.npy", agent.q_table)
                print(f'Game {agent.n_games} | Mean Score: {period_score/100:.2f} | Record: {record} | EXP_RATE: {agent.exploration_rate:.4f} | STEPS: {period_steps/100:.1f}')

                # Guardar en log
                training_log.append([
                    agent.n_games,
                    period_score / 100,
                    agent.exploration_rate
                ])

                record = 0
                period_score = 0
                period_steps = 0
            else:
                training_log.append([
                    agent.n_games,
                    score,
                    agent.exploration_rate
                ])
                

            if score > record:
                record = score

            period_steps += steps
            period_score += score
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            steps = 0

            if agent.n_games == NUM_EPISODES:
                break
        else:
            steps += 1

    # Guardar Q-Table en formato requerido
    q_data = []
    for idx in range(N_STATES):
        velocity, proximity, up_state, down_state, bounces = index_to_state(idx)
        q_values = agent.q_table[idx]
        q_data.append([
            velocity, proximity, up_state, down_state, bounces,
            q_values[0], q_values[1], q_values[2]
        ])

    q_data = np.array(q_data)
    print(q_data)
    np.savetxt("Fernanda-Bley.csv", q_data, delimiter=",", fmt="%.6f")

    # Guardar log de entrenamiento
    import csv
    with open("training_log.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Partida", "Puntaje promedio", "Exploration Rate"])
        writer.writerows(training_log)


if __name__ == '__main__':
    train()