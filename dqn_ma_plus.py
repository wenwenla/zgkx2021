import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from collections import deque

from replay_buffer import ReplayBuffer
from zgkx_env2 import AttackEnv

from entropy_stat import get_rao_quadratic_entropy, get_expected_entropy_over_states, get_det_diversity

import argparse


parser = argparse.ArgumentParser(description='[agents] [epsilon] [boom_step] [boom_range] [boom_cnt] [hit_rate]')
parser.add_argument('--agents', type=int, default=20)
parser.add_argument('--epsilon', type=float, default=1.)
parser.add_argument('--boom_step', type=int, default=20)
parser.add_argument('--boom_range', type=int, default=2)
parser.add_argument('--boom_cnt', type=int, default=2)
parser.add_argument('--hit_rate', type=float, default=1.0)
parser.add_argument('--folder', type=str, default='logs')
parser.add_argument('--history', type=int, default=10)
parser.add_argument('--max_ep', type=int, default=500)
parser.add_argument('--group', type=int, default=1)


args = parser.parse_args()

EPSILON = 0.1
LR = 1e-3
BATCH = 128
GAMMA = 0.95
START_SAMPLES = 2000
REPLAY_BUFFER_SIZE = int(1e6)
RENDER_FLAG = False
FOLDER = args.folder
HISTORY_LEN = args.history
AGENTS = args.agents
MAX_EP = args.max_ep
GROUP_SZ = args.group


class DQNModel(torch.nn.Module):

    def __init__(self, obs_dim, act_cnt):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.q = nn.Linear(64, act_cnt)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_val = self.q(x)
        return q_val


class Policies:

    def __init__(self, obs_dim, act_cnt, policy_state_dict):
        self._model = DQNModel(obs_dim, act_cnt)
        self._model.load_state_dict(policy_state_dict)

    def choose_actions(self, states):
        states = torch.tensor(states).float()
        with torch.no_grad():
            actions = self._model(states).numpy()
        return np.argmax(actions, axis=1)


class DQNAgent:

    def __init__(self, obs_dim, act_cnt):
        self._obs_dim = obs_dim
        self._act_cnt = act_cnt
        self.qnet = DQNModel(self._obs_dim, self._act_cnt)
        self.target_qnet = DQNModel(self._obs_dim, self._act_cnt)
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.optim = optim.Adam(self.qnet.parameters(), lr=LR)
        self.loss_fn = torch.nn.MSELoss()
        self.replay = ReplayBuffer(REPLAY_BUFFER_SIZE, (self._obs_dim, ), 1)
        self._steps = 0

    def get_state_dict(self):
        return self.qnet.state_dict()

    def choose_actions(self, states):
        states = torch.tensor(states).float()
        with torch.no_grad():
            actions = self.qnet(states).numpy()
        return np.argmax(actions, axis=1)

    def choose_action(self, state):
        state = torch.Tensor([state]).float()
        with torch.no_grad():
            action = self.qnet(state).numpy()
        return np.argmax(action)

    def choose_action_with_epsilon(self, state):
        if np.random.uniform() < EPSILON:
            return np.random.randint(0, self._act_cnt)
        return self.choose_action(state)

    def add(self, s, a, r, s_):
        self.replay.add(s, a, r, s_)

    def update(self):
        self._steps += 1
        if self.replay.n_samples() < START_SAMPLES:
            return
        s, a, r, s_ = self.replay.sample(BATCH)
        with torch.no_grad():
            target = self.qnet(torch.Tensor(s))
            nxt_a = self.qnet(torch.Tensor(s_)).argmax(axis=1)
            nxt_q = self.target_qnet(torch.Tensor(s_))
            z = []
            for i, v in enumerate(nxt_a.numpy()):
                z.append(nxt_q[i, v])
            nxt_q = torch.Tensor(z)
            upd = GAMMA * nxt_q
            upd = torch.Tensor(r) + upd
            for i, v in enumerate(a):
                target[i, v] = upd[i]
        self.optim.zero_grad()
        q = self.qnet(torch.Tensor(s))
        loss = self.loss_fn(q, target)
        loss.backward()
        self.optim.step()
        self.soft_copy_parm()
        # if self._steps % 1000 == 0: NO LOG is NEEDED
        #     self.sw.add_scalar('loss/qloss', loss.item(), self._steps)

    def soft_copy_parm(self):
        with torch.no_grad():
            for t, s in zip(self.target_qnet.parameters(), self.qnet.parameters()):
                t.copy_(0.01 * t.data + 0.99 * s.data)

    def parameters(self):
        return self.qnet.state_dict()


def make_env(kwargs):
    return AttackEnv(**kwargs)


class DQNTrainer:

    def __init__(self, env_config):
        self._n_agents = env_config['n_agents']
        self._env = make_env(env_config)

        real_policy = [DQNAgent(self._env.obs_dim, self._env.act_cnt)
                       for _ in range((self._n_agents + GROUP_SZ - 1) // GROUP_SZ)]
        self._policy_mapper = {}
        for i in range(self._n_agents):
            self._policy_mapper[f'uav_{i}'] = real_policy[i // GROUP_SZ]

        self._sampled_states = deque([], maxlen=100000)
        self._policies = [deque([], maxlen=HISTORY_LEN) for _ in range(env_config['n_agents'])]
        self._sw = SummaryWriter(os.path.join(FOLDER, 'logs'))
        self._ep = 0

    def train_one_ep(self):
        self._ep += 1
        states = self._env.reset()

        for sampled_state in states.values():
            if AttackEnv.is_alive(sampled_state):
                self._sampled_states.append(sampled_state)

        done = False
        rew = 0
        while not done:
            agents = self._env.agents_name()
            actions = {}
            for agent in agents:
                p = self._policy_mapper[agent]
                actions[agent] = p.choose_action_with_epsilon(states[agent])

            next_states, rewards, done, info = self._env.step(actions)

            for sampled_state in next_states.values():
                if AttackEnv.is_alive(sampled_state):
                    self._sampled_states.append(sampled_state)

            if RENDER_FLAG:
                self._env.render()

            for agent in agents:
                p = self._policy_mapper[agent]
                p.add(states[agent], actions[agent], rewards[agent], next_states[agent])
                p.update()

            states = next_states
            rew += np.mean(list(rewards.values()))

        for i, agent in enumerate(self._env.agents_name()):
            p = self._policy_mapper[agent]
            self._policies[i].append(Policies(self._env.obs_dim, self._env.act_cnt, p.get_state_dict()))

        indices = np.random.randint(0, len(self._sampled_states), (200, ))
        sampled_states = []
        for p in indices:
            sampled_states.append(self._sampled_states[p])

        rao = self.log_rao_entropy(sampled_states)
        avg_e = self.log_average_entropy(sampled_states)

        self._sw.add_scalar('rewards', rew, self._ep)
        self._sw.add_scalar('rao', rao, self._ep)
        self._sw.add_scalar('avg', avg_e, self._ep)

        if self._ep > 15:
            det_e = self.log_det_diversity(sampled_states)
            self._sw.add_scalar('det', det_e, self._ep)
        else:
            self._sw.add_scalar('det', 0.5, self._ep)  # TODO: HACK

        # self.time_step_log()

        return rew, rao, avg_e

    def log_rao_entropy(self, sampled_states):
        policies = []
        for i in self._env.agents_name():
            policies.append(self._policy_mapper[i])
        return get_rao_quadratic_entropy(policies, sampled_states)

    def log_average_entropy(self, sampled_states):
        policies = []
        for i in self._env.agents_name():
            policies.append(self._policy_mapper[i])
        return get_expected_entropy_over_states(policies, sampled_states)

    def log_det_diversity(self, sampled_states):
        policies = []
        for i in self._env.agents_name():
            policies.append(self._policy_mapper[i])
        return get_det_diversity(policies, sampled_states)

    def time_step_log(self):
        for i in range(self._n_agents):
            rao = get_rao_quadratic_entropy(self._policies[i], self._sampled_states)
            avg_e = get_expected_entropy_over_states(self._policies[i], self._sampled_states)
            self._sw.add_scalar(f'agent_{i}/rao', rao, self._ep)
            self._sw.add_scalar(f'agent_{i}/avg_e', avg_e, self._ep)
            # self._sw.add_scalar(f'agent_{i}/det', avg_e, self._ep)

    def save_model(self):
        torch.save([v.parameters() for v in self._policy_mapper.values()], f'{FOLDER}/{self._ep}.pkl')


def main():
    torch.set_num_threads(1)
    trainer = DQNTrainer({
        'n_agents': AGENTS,
        'attack_epsilon': args.epsilon,
        'boom_step': args.boom_step,
        'boom_range': args.boom_range,
        'boom_cnt': args.boom_cnt,
        'hit_rate': args.hit_rate
    })
    for i in range(MAX_EP):
        r = trainer.train_one_ep()
        print(f'EP: {i} RE: {r}')
        if i % 100 == 0:
            trainer.save_model()


if __name__ == '__main__':
    main()
