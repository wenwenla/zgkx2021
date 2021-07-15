import numpy as np
from PIL import Image
from util import *


class GridEnv:
    OBSTACLE_TOKEN = '*'
    EMTPY_TOKEN = 'o'

    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    MOVE_STOP = 4

    def __init__(self, config, n_agents, max_steps=200, rnd_seed=19971023):
        file_in = open(config, 'r')
        contents = file_in.readlines()
        file_in.close()
        for r in range(len(contents)):
            contents[r] = contents[r].strip()
        self._w = len(contents[0])
        self._h = len(contents)

        print(f'W: {self._w} H: {self._h}')

        self._map = contents
        self._viewer = None

        self.n_agents = n_agents
        self.agent_pos = []
        self.max_steps = max_steps
        self.now_steps = 0
        self.rnd = np.random.RandomState(rnd_seed)

        self.attacker = []

    def get_observations(self):
        # SHOULD BE OVERRIDE BY SUBCLASS
        return {}

    def get_reward(self, prev_pos, actions, next_pos):
        # SHOULD BE OVERRIDE BY SUBCLASS
        return {}

    def update(self):
        # SHOULD BE OVERRIDE BY SUBCLASS
        pass

    def init(self):
        # SHOULD BE OVERRIDE BY SUBCLASS
        pass

    def early_stop(self):
        return False

    def move(self, actions):
        for i in range(self.n_agents):
            if actions[i] == GridEnv.MOVE_UP:
                next_pos = (self.agent_pos[i][0] - 1, self.agent_pos[i][1])
            elif actions[i] == GridEnv.MOVE_DOWN:
                next_pos = (self.agent_pos[i][0] + 1, self.agent_pos[i][1])
            elif actions[i] == GridEnv.MOVE_LEFT:
                next_pos = (self.agent_pos[i][0], self.agent_pos[i][1] - 1)
            elif actions[i] == GridEnv.MOVE_RIGHT:
                next_pos = (self.agent_pos[i][0], self.agent_pos[i][1] + 1)
            else:
                next_pos = self.agent_pos[i]
            if self._map[next_pos[0]][next_pos[1]] == GridEnv.OBSTACLE_TOKEN:
                next_pos = self.agent_pos[i]
            # TODO: notice we may need a quick fix here, two agents may at the same pos
            self.agent_pos[i] = next_pos

    def generate_init_pos(self):
        self.agent_pos = []
        for i in range(self.n_agents):
            pos_r = self.rnd.randint(1, 20)
            pos_c = self.rnd.randint(1, 5)
            self.agent_pos.append((pos_r, pos_c))

    def reset(self):
        self.generate_init_pos()
        self.now_steps = 0
        self.init()
        return self.get_observations()

    def step(self, actions):
        actions_list = []
        for i in range(self.n_agents):  # TODO: fixme
            actions_list.append(actions[f'uav_{i}'])
        actions = actions_list
        assert len(actions) == self.n_agents
        self.now_steps += 1

        prev_pos = self.agent_pos.copy()
        self.move(actions)
        self.update()

        return (self.get_observations(), self.get_reward(prev_pos, actions, self.agent_pos),
                self.now_steps == self.max_steps or self.early_stop(), {})

    def render(self, mode='human'):
        cell_size = 1000 // max(self._w, self._h)
        image = Image.new(mode='RGB', size=(1000, 1000), color='white')
        draw_grid(image, self._h, self._w, 1000 // max(self._h, self._w))

        for r in range(self._h):
            for c in range(self._w):
                if self._map[r][c] == GridEnv.OBSTACLE_TOKEN:
                    draw_rect(image, r, c, cell_size, 'black')

        for pos in self.agent_pos:
            draw_ball(image, pos[0], pos[1], cell_size, 'red')

        for atk in self.attacker:
            draw_ball(image, atk[0], atk[1], cell_size, 'blue')

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
    n_agents = 1
    env = GridEnv('./config2.conf', 1)
    env.reset()
    while True:
        env.step({f'uav_{k}': np.random.randint(0, 5) for k in range(n_agents)})
        env.render()


if __name__ == '__main__':
    main()
