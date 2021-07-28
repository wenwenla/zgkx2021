import copy
import itertools
import math
import time

import torch

from util import *
from PIL import Image

import numpy as np


def get_dist(point0, point1):
    return abs(point1[0] - point0[0]) + abs(point1[1] - point0[1])


class AttackEnv:

    MAX_SIZE_ROW = 20
    MAX_SIZE_COL = 20
    VIEW_RANGE = 3
    MISSION_COMPLETE_REWARD = 150
    ATTACK_PENALTY = 100

    @staticmethod
    def is_alive(state):
        return abs(state[-3] - 1.0) < 1e-6

    def __init__(self, n_agents, attack_epsilon, boom_step, boom_range, boom_cnt, hit_rate):
        self._n_agents = n_agents
        self._pos = [
            [0, 0] for _ in range(n_agents)
        ]
        self._alive = [
            True for _ in range(n_agents)
        ]
        self._attack_epsilon = attack_epsilon
        self._boom_step = boom_step
        self._boom_range = boom_range
        self._boom_cnt = boom_cnt
        self._step = 0
        self._boom_areas = []
        self._bit_cnt = None
        self._hit_rate = hit_rate
        self._viewer = None
        self._mission_completed = True

        self.obs_dim = (2 * AttackEnv.VIEW_RANGE + 1) ** 2 + 3
        self.act_cnt = 5
        self.n_agents = n_agents

    def agents_name(self):
        return [f'uav_{i}' for i in range(self._n_agents)]

    def get_observation(self):
        # 0: empty, 1: friends, 2: boom
        target_position = [AttackEnv.MAX_SIZE_ROW - 1, AttackEnv.MAX_SIZE_COL - 1]
        bitmap = np.zeros((AttackEnv.MAX_SIZE_ROW, AttackEnv.MAX_SIZE_COL))
        for i in range(self._n_agents):
            bitmap[self._pos[i][0], self._pos[i][1]] = 1
        for bp in self._boom_areas:
            bitmap[bp[0]: bp[0] + self._boom_range, bp[1]: bp[1] + self._boom_range] = 2
        bitmap = np.pad(bitmap, AttackEnv.VIEW_RANGE, 'constant', constant_values=-1)
        obs = {}
        for i in range(self._n_agents):
            code_len = (2 * AttackEnv.VIEW_RANGE + 1) ** 2
            now_view = bitmap[self._pos[i][0]: self._pos[i][0] + 2 * AttackEnv.VIEW_RANGE + 1,
                       self._pos[i][1]: self._pos[i][1] + 2 * AttackEnv.VIEW_RANGE + 1].reshape((code_len, ))
            obs[f'uav_{i}'] = np.hstack([now_view, [int(self._alive[i] or self._pos[i] == target_position), self._pos[i][0], self._pos[i][1]]])
        return obs

    def get_reward(self, prev_positions, now_positions):
        rewards = [0 for _ in range(self._n_agents)]
        target_position = [AttackEnv.MAX_SIZE_ROW - 1, AttackEnv.MAX_SIZE_COL - 1]
        # distance based reward
        for i in range(self._n_agents):
            dist_prev = get_dist(prev_positions[i], target_position)
            dist_next = get_dist(now_positions[i], target_position)
            rewards[i] += dist_prev - dist_next
        # attack based reward
        atk_list = []

        for i in range(self._n_agents):
            attacked = False
            for j in range(len(self._boom_areas)):
                if self._boom_areas[j][0] <= self._pos[i][0] < self._boom_areas[j][0] + self._boom_range and \
                        self._boom_areas[j][1] <= self._pos[i][1] < self._boom_areas[j][1] + self._boom_range:
                    attacked = True
                    break
            if attacked and np.random.uniform() < self._hit_rate:
                rewards[i] -= AttackEnv.ATTACK_PENALTY
                atk_list.append(i)

        # time based reward
        for i in range(self._n_agents):
            rewards[i] -= 1

        for i in range(self._n_agents):
            if now_positions[i] == target_position:
                for j in range(self._n_agents):
                    # if one UAV reaches the target area,
                    # the task is completed and all UAVs will be rewarded by `MISSION_COMPLETE_REWARD`
                    rewards[j] = AttackEnv.MISSION_COMPLETE_REWARD
                self._mission_completed = True
                break

        if not self._mission_completed:
            for i in range(self._n_agents):
                if not self._alive[i]:
                    rewards[i] = 0

        for atk in atk_list:
            self._alive[atk] = False

        return {
            f'uav_{i}': rewards[i] for i in range(self._n_agents)
        }

    def reset(self):
        self._step = 0
        self._pos = [
            [0, 0] for _ in range(self._n_agents)
        ]
        self._alive = [
            True for _ in range(self._n_agents)
        ]
        self._mission_completed = False
        return self.get_observation()

    def step(self, actions):
        dirs = [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]]

        prev_pos = copy.deepcopy(self._pos)

        for i, uav_name in enumerate(self.agents_name()):
            if not self._alive[i]:
                continue
            if self._pos[i][0] == AttackEnv.MAX_SIZE_ROW - 1 and self._pos[i][1] == AttackEnv.MAX_SIZE_COL - 1:
                continue

            if 0 <= self._pos[i][0] + dirs[actions[uav_name]][0] < AttackEnv.MAX_SIZE_ROW:
                self._pos[i][0] += dirs[actions[uav_name]][0]

            if 0 <= self._pos[i][1] + dirs[actions[uav_name]][1] < AttackEnv.MAX_SIZE_COL:
                self._pos[i][1] += dirs[actions[uav_name]][1]

        self._bit_cnt = np.zeros((AttackEnv.MAX_SIZE_ROW, AttackEnv.MAX_SIZE_COL), int)

        for i, p in enumerate(self._pos):
            if self._alive[i] and 1 <= p[1] < AttackEnv.MAX_SIZE_COL - 1:
                self._bit_cnt[p[0], p[1]] += 1

        if self._step % self._boom_step == 0:
            # generate boom areas
            self._boom_areas.clear()

            attack_choices = []
            for row in range(0, AttackEnv.MAX_SIZE_ROW - self._boom_range + 1):
                for col in range(4, AttackEnv.MAX_SIZE_COL - self._boom_range - 1 + 1):
                    cnt = 0
                    for dr, dc in itertools.product(range(self._boom_range), range(self._boom_range)):
                        cnt += self._bit_cnt[row + dr, col + dc]
                    attack_choices.append((cnt, row, col))

            attack_choices.sort(key=lambda x: (-x[0], x[1], x[2]))
            attack_choices_index = 0
            for i in range(self._boom_cnt):
                if np.random.uniform() < self._attack_epsilon:
                    self._boom_areas.append(
                        [
                            np.random.randint(0, AttackEnv.MAX_SIZE_ROW - self._boom_range),
                            np.random.randint(1, AttackEnv.MAX_SIZE_COL - self._boom_range - 1),
                        ]
                    )
                else:
                    self._boom_areas.append(
                        [
                            attack_choices[attack_choices_index][1],
                            attack_choices[attack_choices_index][2]
                        ]
                    )
                    attack_choices_index += 1

        self._step += 1

        obs = self.get_observation()
        rew = self.get_reward(prev_pos, self._pos)

        all_die = True
        for i in range(self._n_agents):
            if self._alive[i] and self._pos[i] != [AttackEnv.MAX_SIZE_ROW - 1, AttackEnv.MAX_SIZE_COL - 1]:
                all_die = False
                break

        return obs, rew, self._step == 200 or all_die or self._mission_completed, {}

    def render(self, mode='human'):
        cell_size = 1000 // max(AttackEnv.MAX_SIZE_ROW, AttackEnv.MAX_SIZE_COL)
        image = Image.new(mode='RGB', size=(1000, 1000), color='white')
        draw_grid(image, AttackEnv.MAX_SIZE_ROW, AttackEnv.MAX_SIZE_COL, 1000 // max(AttackEnv.MAX_SIZE_ROW, AttackEnv.MAX_SIZE_COL))

        for pos in self._boom_areas:
            for dr, dc in itertools.product(range(self._boom_range), range(self._boom_range)):
                draw_ball(image, pos[0] + dr, pos[1] + dc, cell_size, 'red')

        for i, pos in enumerate(self._pos):
            draw_ball(image, pos[0], pos[1], cell_size, 'blue' if self._alive[i] else 'black')

        image = np.asarray(image)
        if mode == 'rgb_array':
            return image
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self._viewer is None:
                self._viewer = rendering.SimpleImageViewer()
            self._viewer.imshow(image)
            return self._viewer.isopen


def main():
    env = AttackEnv(n_agents=20, attack_epsilon=1.0, boom_step=10, boom_range=2, boom_cnt=2, hit_rate=0.5)
    env.reset()
    done = False
    pre = time.time()

    while not done:
        _, _, done, _ = env.step({
            f'uav_{i}': np.random.randint(0, 5) for i in range(env._n_agents)
        })

        env.render()
        # time.sleep(0.2)
    nxt = time.time()
    print(f'Time: {nxt - pre}')


if __name__ == '__main__':
    main()
