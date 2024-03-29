import math
from collections import defaultdict
import numpy as np


def information_entropy(acts):
    cnt = defaultdict(float)
    for a in acts:
        cnt[a] += 1
    for k in cnt.keys():
        cnt[k] /= len(acts)
    result = 0
    for v in cnt.values():
        result -= v * math.log(v)
    return result


def get_rao_quadratic_entropy(policies, sampled_states):
    acts = []
    for v in policies:
        acts.append(v.choose_actions(sampled_states))
    policy_index = 0
    i_p = {}
    p_i = {}
    policies_set = set()
    for i, a in enumerate(acts):
        hash_index = tuple(a.tolist())
        if hash_index not in policies_set:
            i_p[policy_index] = a
            p_i[hash_index] = 1
            policy_index += 1
            policies_set.add(hash_index)
        else:
            p_i[hash_index] += 1

    dist = np.zeros((policy_index, policy_index))

    for i in range(policy_index):
        for j in range(i + 1, policy_index):
            policy_i = i_p[i]
            policy_j = i_p[j]
            d = np.sum(policy_i != policy_j) / len(sampled_states)
            dist[i, j] = d
            dist[j, i] = d
    policy_prob = np.zeros((policy_index, ), dtype='float')

    for i in range(policy_index):
        policy_prob[i] = p_i[tuple(i_p[i].tolist())] / len(policies)

    result = 0
    for i in range(policy_index):
        for j in range(policy_index):
            result += dist[i, j] * policy_prob[i] * policy_prob[j]
    return result


def get_expected_entropy_over_states(policies, sampled_states):
    acts = []
    for v in policies:
        acts.append(v.choose_actions(sampled_states))
    total_entropy = 0
    for i in range(len(sampled_states)):
        actions_on_this_state = []
        for p in range(len(policies)):
            actions_on_this_state.append(acts[p][i])
        e = information_entropy(actions_on_this_state)
        total_entropy += e
    total_entropy /= len(sampled_states)
    return total_entropy


def binary_det(acts, n):
    target = 0.5
    ll = 1e-8
    lr = 100
    while lr - ll > 1e-8:
        mid = (ll + lr) / 2
        matrix = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                l = 2 * mid * mid
                d = np.exp(-(np.linalg.norm((acts[i] - acts[j] != 0).astype(float), 2) ** 2) / l)
                matrix[i, j] = d
                matrix[j, i] = d
        div = np.linalg.det(matrix)
        if div > target:
            ll = mid
        else:
            lr = mid
    return lr


DET_L = None


def get_det_diversity(policies, sampled_states):
    global DET_L
    acts = []
    n = len(policies)
    for v in policies:
        acts.append(v.choose_actions(sampled_states))

    if DET_L is None:
        DET_L = binary_det(acts, n)
        print(f'DET_L: {DET_L} N_AGENTS: {n}')

    matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            l = 2 * DET_L * DET_L
            d = np.exp(-(np.linalg.norm((acts[i] - acts[j] != 0).astype(float), 2) ** 2) / l)
            matrix[i, j] = d
            matrix[j, i] = d
    div = np.linalg.det(matrix)
    return div
