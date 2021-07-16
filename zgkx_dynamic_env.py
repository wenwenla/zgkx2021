import copy
import time

import numpy as np
from gridenv import GridEnv


class DynamicEnv(GridEnv):

    def __init__(self, n_agents):
        super().__init__('./config2.conf', n_agents)
        self.obs_dim = 14
        self.act_cnt = 5

    def agents_name(self):
        return [f'uav_{index}' for index in range(self.n_agents)]

    def init(self):
        self.attacker = [[18, 25], [1, 25]]

    def get_observations(self):
        result = {}
        dx = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]

        attacker_pos = set()
        for p in self.attacker:
            attacker_pos.add(tuple(p))

        for index, pos in enumerate(self.agent_pos):
            obs = [pos[0], pos[1]]
            for r, c in dx:
                n = (pos[0] + r, pos[1] + c)
                if n[0] < 0 or n[0] >= len(self._map) or n[1] < 0 or n[1] >= len(self._map[0]) \
                        or self._map[n[0]][n[1]] == GridEnv.OBSTACLE_TOKEN:
                    obs.append(0)
                elif n in attacker_pos:
                    obs.append(0.5)
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

        attacker_set = set()
        for a in self.attacker:
            attacker_set.add(tuple(a))

        for i, p in enumerate(next_pos):
            if p in attacker_set:
                result[f'uav_{i}'] -= 0.1
        return result

    def update(self):
        used = set()

        for i, a in enumerate(self.attacker):
            min_dist = 1e9
            uav_p = None
            for uav in self.agent_pos:
                dist = abs(a[0] - uav[0]) + abs(a[1] - uav[1])
                if dist < min_dist:
                    min_dist = dist
                    uav_p = uav
            dirs = [[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]]
            now_d = 1e9
            now_p = None
            for d in dirs:
                tmp_p = [a[0] + d[0], a[1] + d[1]]
                if tmp_p[0] < 1 or tmp_p[0] > 20 or \
                        tmp_p[1] < 12 or tmp_p[1] > 34:
                    continue
                if self._map[tmp_p[0]][tmp_p[1]] == GridEnv.OBSTACLE_TOKEN:
                    continue
                if tuple(tmp_p) in used:
                    continue
                dist = abs(tmp_p[0] - uav_p[0]) + abs(tmp_p[1] - uav_p[1])
                if dist < now_d:
                    now_d = dist
                    now_p = tmp_p
            if now_p:
                self.attacker[i] = now_p
            used.add(tuple(self.attacker[i]))


def make_env(**kwargs):
    return DynamicEnv(kwargs['n_agents'])


def main():
    env = DynamicEnv(20)
    env.reset()
    while True:
        r = env.step({f'uav_{k}': np.random.randint(0, 5) for k in range(20)})
        print(r)
        env.render()
        time.sleep(0.1)


if __name__ == '__main__':
    main()
