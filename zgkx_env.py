import time

import numpy as np
from gridenv import GridEnv


class EnvEasy(GridEnv):

    def __init__(self, n_agents):
        super().__init__('./config2.conf', n_agents)

    def agents_name(self):
        return [f'uav_{index}' for index in range(self.n_agents)]

    def get_observations(self):
        result = {}
        dx = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for index, pos in enumerate(self.agent_pos):
            obs = [pos[0], pos[1]]
            for r, c in dx:
                n = (pos[0] + r, pos[1] + c)
                if self._map[n[0]][n[0]] == GridEnv.OBSTACLE_TOKEN:
                    obs.append(0)
                else:
                    obs.append(1)
            result[f'uav_{index}'] = obs
        return result

    def get_reward(self, prev_pos, actions, next_pos):
        result = {f'uav_{k}': 0 for k in range(self.n_agents)}
        for i, p in enumerate(next_pos):
            if 11 <= p[0] <= 20 and 36 <= p[1] <= 45:
                result[f'uav_{i}'] += 1
            else:
                result[f'uav_{i}'] -= 1

        for i in range(len(prev_pos)):
            dist_prev = abs(prev_pos[i][0] - 15) + abs(prev_pos[i][1] - 40)
            dist_next = abs(next_pos[i][0] - 15) + abs(next_pos[i][1] - 40)
            potential = dist_prev - dist_next
            result[f'uav_{i}'] += potential

        return result


def make_env(**kwargs):
    return EnvEasy(kwargs['n_agents'])


def main():
    env = EnvEasy(20)
    env.reset()
    while True:
        r = env.step({f'uav_{k}': np.random.randint(0, 5) for k in range(2)})
        print(r)
        env.render()
        time.sleep(1)


if __name__ == '__main__':
    main()
