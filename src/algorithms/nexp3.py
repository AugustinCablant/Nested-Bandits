""" 
Same file as NEW but with a different method sample_node_path corresponding to NEXP3
"""

import os
import sys
base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

import numpy as np
from tqdm import tqdm
import time
import psutil
from src.utils.save_results import save_result

EPS = 1e-8

class NEXP3:

    def __init__(self, settings):
        """
        :param number_of_actions: number of actions from which the slates will be formed, K.
        :param slate_size: slate size, s.
        :param max_rounds: the number of rounds for which the algorithm will run.
        """

        self.rng = np.random.RandomState(settings['rd'])
        self.max_round = settings['max_rounds']
        self.settings = settings
        self.lr = 1

    def set_environment(self, environment):
        """
        :param environment: this should be a function that can take a vector of size K
        (indicator vector of the chosen slate), and the current round, t as parameters and return the loss/reward
        associated with that slate and that slate only. The indicator vector will have non-zero elements which represent
        the chosen actions in that slate and zero elements which represent actions that are not chosen. The reward/loss
        for actions that are not chosen must be 0, and for the chosen actions the reward/loss should be in [-1,1] or else
        it will be clipped. Hence the output vector must also be a vector of size K with elements clipped to be in [-1,1].
        """
        self.environment = environment
        self.leaves = self.environment.tree.get_all_leaves()
        self.nb_leaves_per_class = self.environment.nb_leaves_per_class
        self.nb_levels = self.environment.nb_levels
        self.K = self.nb_leaves_per_class ** (self.nb_levels-1)
        self.proba = np.full(self.K, 1.0 / self.K)

    def vector_proba(self, y):
        """

        Parameters
        ----------
        y

        Returns
        -------

        """
        stable_exp_y = np.exp(y - np.max(y))
        proba_vector = stable_exp_y/np.sum(stable_exp_y)
        return proba_vector

    def sample_action(self, round):
        self.lr = 1 / np.sqrt(round+1)
        leaves = self.environment.tree.get_all_leaves()
        a_t = self.rng.choice(np.range(self.K), 
                              p = self.proba)
        leaves_name = [leaf.name for leaf in self.leaves]

        node_path = []
        proba_path = []
        reward_path = []
        node = leaves[a_t]
        reward_path.append(self.environment.get_reward_by_node(node))
        node_path.append(a_t)
        proba_path.append(self.proba[a_t])

        past_vector_proba = self.proba
        past_indexes = np.range(self.K)
        past_nodes = self.leaves
        past_names = [leaf.name for leaf in past_nodes]

        while bool(node.parent):
            parent = node.parent

            current_nodes = list(set([node.parent for node in past_nodes]))
            current_names = [n.name for n in current_nodes]
            current_indexes = len(current_nodes)
            current_vector_proba = np.zeros(current_indexes)
            for i,node in enumerate(current_nodes):
                childrens_node = node.children
                childrens_name = [c.name for c in childrens_node]
                indexes = [i for i,past_name in enumerate(past_names) if past_name in childrens_name]
                current_vector_proba[i] = np.sum(past_vector_proba[indexes])

            index_parent = list(current_names).index(parent.name)
            proba_path.append(current_vector_proba[current_vector_proba[index_parent]])
            

            node_path.append(parent) 
            reward_parent = self.environment.get_reward_by_node(parent)
            reward_path.append(reward_parent)
            node = parent
            past_vector_proba = current_vector_proba
            past_indexes = current_indexes
            past_nodes = current_nodes
            past_names = current_names

        node_path = node_path[::-1] 
        reward_path = reward_path[::-1]
        proba_path = proba_path[::-1]
        
        return a_t, node_path, reward_path, proba_path


    def update_probas(self, a_t, proba_path, reward_path):
        #Backward routine
        G = 0
        for p, loss in zip(proba_path, reward_path):
            G += loss / p

        self.proba[a_t] *= np.exp(-self.lr*G)
        self.proba = self.vector_proba(self.proba)

    def iterate_learning(self):
        """
        run the agent for "max_rounds" rounds
        """
        metrics = {
            'reward': [],
            'regret': [],
            'round': []
        }
        regrets = []
        rewards = []

        for round in tqdm(range(0, self.max_round)):

            # Choose action and receive reward iteratively
            a_t, node_path, reward_path, proba_path = self.sample_action(round)

            # Reward from environment
            reward = np.sum(reward_path)
            best_strategy_reward = self.environment.get_best_strategy_reward()
            regrets.append(best_strategy_reward - reward)
            rewards.append(reward)

            # Update probas
            self.update_probas(a_t, proba_path, reward_path)

            if round % 100 == 0:
                metrics['reward'].append(np.mean(rewards))
                regret = np.sum(regrets)
                metrics['regret'].append(regret)
                metrics['round'].append(round)
                save_result(self.settings, regret, np.mean(rewards), round)

        # Visualization
        self.score_vector = None

        return metrics