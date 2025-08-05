""" 
File for NTsallis-Inf
"""

import os
import sys
base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)
from scipy.optimize import minimize
import numpy as np
from tqdm import tqdm
import time
import psutil
from src.utils.save_results import save_result

EPS = 1e-12

class NTSALLISINF:
    def __init__(self, settings):
        self.rng = np.random.RandomState(settings['rd'])
        self.max_round = settings['max_rounds']
        self.settings = settings
        self.lr = 1
        

    def set_environment(self, environment):
        self.environment = environment
        self.leaves = self.environment.tree.get_all_leaves()
        self.nb_leaves_per_class = self.environment.nb_leaves_per_class
        self.nb_levels = self.environment.nb_levels
        self.K = len(self.leaves)
        self.proba = np.full(self.K, 1.0 / self.K)
    
    
    def sample_leaf(self, round):
        self.lr = 1 / np.sqrt(round + 1)
        #Select a leaf according to p_t
        a_t = self.rng.choice(range(self.K), p=self.proba)
        node_selected = self.leaves[a_t]
        node_path = list(self.environment.tree.get_parent_nodes(node_selected))
        node_path.append(node_selected)
        node_path = node_path[1:] #eject the root

        #Obtain the rewards and the corresponding nodes
        reward_path = [self.environment.get_reward_by_node(node) for node in node_path]

        proba_path = self.obtain_proba_path(a_t)

        node_path.reverse()
        reward_path.reverse()
        proba_path.reverse()

        return a_t, node_path, reward_path, proba_path

    def obtain_proba_path(self, a_t):
        proba_path = [self.proba[a_t]]
        node_selected = self.leaves[a_t]
        node = node_selected
        while node.parent.name != "root":
            node = node.parent
            node_childrens = node.children
            node_childrens_names = [n.name for n in node_childrens]
            indexes = [i for i,node in enumerate(self.leaves) if node.name in node_childrens_names]
            proba_path.append(np.sum(self.proba[indexes]))
        return proba_path


    def update_probas(self, a_t, proba_path, reward_path):
        G = sum(loss / (p + EPS) for p, loss in zip(proba_path, reward_path))

        def tsallis_inf_divergence(p, q):
            """ Tsallis Divergence for q = 1/2 """
            p = np.array(p, dtype=np.float64)
            q = np.array(q, dtype=np.float64)

            if not np.isclose(p.sum(), 1.0):
                p /= p.sum()
            if not np.isclose(q.sum(), 1.0):
                q /= q.sum()
                
            return 2 * np.sum(np.sqrt(p) - np.sqrt(q)) - np.sum((p-q)/np.sqrt(p))

        pt = self.proba.copy()

        def tsallis_mirror_obj(p):
            return (
                self.lr * G * (p[a_t] - pt[a_t]) + tsallis_inf_divergence(p, pt)
                 )
        
        constraints = [{'type': 'eq', 'fun': lambda p: np.sum(p) - 1}]
        bounds = [(0, 1) for _ in range(self.K)]
        x0 = pt.copy()
        result = minimize(
                        tsallis_mirror_obj,
                        x0,
                        bounds=bounds,
                        constraints=constraints,
                        method='SLSQP',
                        options={'ftol': 1e-10, 'maxiter': 500}
                        )
        if result.success and np.all(result.x >= 0):
            self.proba = result.x
            #self.proba = p / np.sum(p)
        else:
            self.proba = np.ones(self.K) / self.K
        

    def iterate_learning(self):
        metrics = {
            'reward': [],
            'regret': [],
            'round': []
        }
        regrets = []
        rewards = []

        for round in tqdm(range(self.max_round)):
            a_t, node_path, reward_path, proba_path = self.sample_leaf(round)
            reward = np.sum(reward_path)
            best_strategy_reward = self.environment.get_best_strategy_reward()
            regrets.append(best_strategy_reward - reward)
            rewards.append(reward)
            self.update_probas(a_t, proba_path, reward_path)
            if round % 100 == 0:
                mean_reward = np.mean(rewards)
                total_regret = np.sum(regrets)
                metrics['reward'].append(mean_reward)
                metrics['regret'].append(total_regret)
                metrics['round'].append(round)
                save_result(self.settings, total_regret, mean_reward, round)

        self.score_vector = None
        return metrics