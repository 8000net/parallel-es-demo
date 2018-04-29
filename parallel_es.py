import multiprocessing
import sys

import gym
import numpy as np

from es import ES

POP_SIZE = 16 # Number of solutions in each generation
STDDEV = 1.0
FITNESS_GOAL = 195
NUM_WORKERS = POP_SIZE
N_ROLLOUTS_PER_TRIAL = 100

class Controller:
    def __init__(self, parameters, input_dim=4, hidden_units=50, output_dim=1):
        self.W1 = np.reshape(parameters[:input_dim*hidden_units],
                            (hidden_units, input_dim))
        self.W2 = np.reshape(parameters[input_dim*hidden_units:],
                             (output_dim, hidden_units))

    def get_action(self, x):
        z = np.dot(self.W1, x)
        a = np.tanh(z)
        z = np.dot(self.W2, a)
        return int(np.heaviside(z, 0)[0])


def rollouts(agent, env, n=N_ROLLOUTS_PER_TRIAL):
    avg_reward = 0
    for i in range(n):
        obs = env.reset()
        done = False
        total_reward = 0
        t = 1
        while not done:
            a = agent.get_action(obs)
            obs, reward, done, info = env.step(a)
            env.render()
            total_reward += reward
            t += 1

        avg_reward += total_reward

    env.close()
    avg_reward /= n
    print('Avg reward (over %d rollouts): %d' % (n, avg_reward))
    return avg_reward


def start_work(env, solution):
    controller = Controller(solution)
    return rollouts(controller, env)


def train():
    solver = ES(pop_size=POP_SIZE, n_dim=250, stddev=1.0)
    envs = [gym.make('CartPole-v0')
            for _ in range(NUM_WORKERS)]

    pool = multiprocessing.Pool(processes=NUM_WORKERS)

    gen = 0
    while True:
        print('Generation %d' % gen)
        solutions = solver.ask()
        fitness_list = np.zeros(POP_SIZE)


        worker_results = [pool.apply_async(start_work, (envs[i], solutions[i]))
                          for i in range(POP_SIZE)]

        fitness_list = [res.get() for res in worker_results]

        solver.tell(fitness_list)
        best_solution, best_fitness = solver.result()

        print('Generation %d: chose best solution with fitness %f' % (
            gen, best_fitness))

        if best_fitness >= FITNESS_GOAL:
            np.save('controller-params.npy', best_solution)
            break

        gen += 1

    for env in envs:
        env.close()


def test():
    envs = [gym.make('CartPole-v0')
            for _ in range(NUM_WORKERS)]

    pool = multiprocessing.Pool(processes=NUM_WORKERS)

    solution = np.load('./controller-params.npy')

    worker_results = [pool.apply_async(start_work, (envs[i], solution))
                      for i in range(POP_SIZE)]

    fitness_list = [res.get() for res in worker_results]

    for env in envs:
        env.close()


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train()

    if sys.argv[1] == 'test':
        test()
