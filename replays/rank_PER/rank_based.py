#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description: 

import sys
import math
import random
import numpy as np
from . import binary_heap
from replays.base_replay import BaseReplay


class RankPER(BaseReplay):

    def __init__(self,max_size,batch_size,alpha=0.7,beta =0.7):
        super().__init__(max_size,batch_size)
        self.alpha = alpha

        self.replace_flag = True
        self.priority_size = self.max_size

        self.alpha = alpha
        self.beta = beta
        self.batch_size = batch_size
        # self.total_steps = conf['steps'] if 'steps' in conf else 100000

        # partition number N, split total size to N part
        self.partition_num = 100

        self.index = 0
        self.isFull = False

        self.buffer = {}
        self.priority_queue = binary_heap.BinaryHeap(self.priority_size)
        self.distributions = self.build_distributions()

        # self.beta_grad = (1 - self.beta_zero) / float(self.total_steps - self.learn_start)

    def max_priority(self):
        return self.priority_queue.get_max_priority()

    def build_distributions(self):
        """
        preprocess pow of rank
        (rank i) ^ (-alpha) / sum ((rank i) ^ (-alpha))
        :return: distributions, dict
        """
        res = {}
        n_partitions = self.partition_num
        partition_num = 1
        # each part size
        partition_size = int(math.floor(self.max_size / n_partitions))

        for n in range(partition_size, self.max_size + 1, partition_size):
            if 0 <= n <= self.priority_size:
                distribution = {}
                # P(i) = (rank i) ^ (-alpha) / sum ((rank i) ^ (-alpha))
                pdf = list(
                    map(lambda x: math.pow(x, -self.alpha), range(1, n + 1))
                )
                pdf_sum = math.fsum(pdf)
                distribution['pdf'] = list(map(lambda x: x / pdf_sum, pdf))
                # split to k segment, and than uniform sample in each k
                # set k = batch_size, each segment has total probability is 1 / batch_size
                # strata_ends keep each segment start pos and end pos
                cdf = np.cumsum(distribution['pdf'])
                strata_ends = {1: 1, self.batch_size + 1: n}
                step = 1 / float(self.batch_size)
                index = 1
                for s in range(2, self.batch_size + 1):
                    while cdf[index] < step:
                        index += 1
                    strata_ends[s] = index
                    step += 1 / float(self.batch_size)

                distribution['strata_ends'] = strata_ends

                res[partition_num] = distribution

            partition_num += 1

        return res

    def fix_index(self):
        """
        get next insert index
        :return: index, int
        """
        if self.size <= self.max_size:
            self.size += 1
        if self.index % self.max_size == 0:
            self.isFull = True if len(self.buffer) == self.max_size else False
            if self.replace_flag:
                self.index = 1
                return self.index
            else:
                sys.stderr.write('Experience replay buff is full and replace is set to FALSE!\n')
                return -1
        else:
            self.index += 1
            return self.index

    def add(self, data, priority):
        """
        store experience, suggest that experience is a tuple of (s1, a, r, s2, t)
        so each experience is valid
        :param experience: maybe a tuple, or list
        :return: bool, indicate insert status
        """
        insert_index = self.fix_index()
        assert insert_index > 0
        if insert_index in self.buffer:
            del self.buffer[insert_index]
        self.buffer[insert_index] = data
        # add to priority queue
        priority = self.priority_queue.get_max_priority()
        self.priority_queue.update(priority, insert_index)
        return True

    def retrieve(self, indices):
        """
        get experience from indices
        :param indices: list of experience id
        :return: experience replay sample
        """
        return [self.buffer[v] for v in indices]

    def rebalance(self):
        """
        rebalance priority queue
        :return: None
        """
        self.priority_queue.balance_tree()

    def priority_update(self, indices, priorities):
        """
        update priority according indices and deltas
        :param indices: list of experience id
        :param delta: list of delta, order correspond to indices
        :return: None
        """
        for i in range(0, len(indices)):
            self.priority_queue.update(math.fabs(delta[i]), indices[i])

    def sample(self):
        """
        sample a mini batch from experience replay
        :param global_step: now training step
        :return: experience, list, samples
        :return: w, list, weights
        :return: rank_e_id, list, samples id, used for update priority
        """
        if self.size <= self.batch_size * 10:
            return None
        dist_index = math.floor(self.size / self.max_size * self.partition_num)
        # issue 1 by @camigord
        partition_size = math.floor(self.max_size / self.partition_num)
        partition_max = dist_index * partition_size
        distribution = self.distributions[dist_index]
        rank_list = []
        # sample from k segments
        for n in range(1, self.batch_size + 1):
            index = random.randint(distribution['strata_ends'][n],
                                   distribution['strata_ends'][n + 1])
            rank_list.append(index)

        # find all alpha pow, notice that pdf is a list, start from 0
        alpha_pow = [distribution['pdf'][v - 1] for v in rank_list]
        # w = (N * P(i)) ^ (-beta) / max w
        weights = np.power(np.array(alpha_pow) * partition_max, -self.beta)
        w_max = max(weights)
        weights = np.divide(weights, w_max)
        # rank list is priority id
        # convert to experience id
        indices = self.priority_queue.priority_to_experience(rank_list)
        # get experience id according rank_e_id

        state, action, reward, next_state, done = [], [], [], [], []

        for i in indices:
            s, a, r, s_, d = self.buffer[i]
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            next_state.append(np.array(s_, copy=False))
            done.append(np.array(d, copy=False))

        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done), weights, indices
