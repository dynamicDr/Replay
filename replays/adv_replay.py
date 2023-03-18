import math
from collections import deque
import sys
sys.path.append("..")
sys.path.append(".")
import numpy
import random

import numpy as np
import torch

from replays.base_replay import BaseReplay
from replays.proportional_PER import sum_tree

class AdvPER(BaseReplay):

    def __init__(self,max_size,batch_size,alpha=0.7,beta = 0.7,verbose=False):
        super().__init__(max_size,batch_size)
        self.verbose = verbose
        self.stage = 1

        self.new_buffer = sum_tree.SumTree(self.max_size)
        self.old_buffer = sum_tree.SumTree(self.max_size)

        self.alpha = alpha
        self.beta = beta
        self.max_p = 1.0
        self.saved_critic = None
        self.lamda = 1.0

        self.sample_from_new = self.batch_size
        self.sample_from_old = 0

    def update_saved_critic(self,critic):
        self.saved_critic = critic

    def max_priority(self):
        return self.max_p

    def check_stage_change(self):
        # TODO
        if self.stage == 1:
            pass
        elif self.stage == 2:
            pass

    def stage_1_to_2(self):
        self.old_buffer = self.new_buffer
        self.new_buffer = sum_tree.SumTree(self.max_size)
        self.lamda = 1.0
        self.stage = 2

    def stage_2_to_1(self):
        self.stage = 1
        self.max_p = 1.0
        priorities = [self.max_p for _ in range(self.new_buffer.filled_size())]
        self.priority_update(self.new_buffer, range(self.new_buffer.filled_size()), priorities)

    def recalculate_lamda(self):
        if self.stage ==1 :
            return
        if self.old_buffer == 0:
            self.stage_2_to_1()
        proportion = self.new_buffer.size / (self.old_buffer.size + self.new_buffer.size)
        if self.lamda > proportion:
            # TODO decay lamda
            self.lamda -= 0.01
        else:
            self.lamda = proportion
        self.sample_from_new = math.floor(self.lamda * self.batch_size)
        self.sample_from_old = self.batch_size - self.sample_from_new

    def add(self, data, priority=None):
        if priority is None:
            priority = self.max_priority()
        self.new_buffer.add(data, priority)
        if self.stage == 2 and self.new_buffer.size + self.old_buffer.size >= self.max_size:
            self.old_buffer.remove()

    def _sample_from(self,buffer,sample_size):
        indices = []
        weights = []
        priorities = []
        state, action, reward, next_state, done = [], [], [], [], []
        for _ in range(sample_size):
            r = random.uniform(0, 1)
            data, priority, index = buffer.find(r)
            priorities.append(priority)
            weights.append((1. / self.max_size / priority) ** self.beta if priority > 1e-16 else 0)
            indices.append(index)
            self.priority_update(buffer, [index], [0])  # To avoid duplicating
            s, a, r, s_, d = data
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            next_state.append(np.array(s_, copy=False))
            done.append(np.array(d, copy=False))
        self.priority_update(buffer, indices, priorities)  # Revert priorities
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done),weights,indices

    def sample(self):
        if self.stage == 1:
            return self._sample_from(self.new_buffer,self.batch_size)
        else:
            state, action, reward, next_state, done ,weights, indices = self._sample_from(self.new_buffer, self.sample_from_new)
            if self.sample_from_old !=0:
                t_state, t_action, t_reward, t_next_state, t_done, t_weights, t_indices = self._sample_from(self.old_buffer, self.sample_from_old)
                state = np.concatenate((state, t_state))
                action = np.concatenate((action, t_action))
                reward = np.concatenate((reward, t_reward))
                next_state = np.concatenate((next_state, t_next_state))
                done = np.concatenate((done, t_done))
                weights = np.concatenate((weights, t_weights))
                indices = np.concatenate((indices, t_indices))
            return state, action, reward, next_state, done, weights, indices

    def get_q(self, indices, Q, next_state, next_action,reward,done,gamma,writer,episode):
        if self.stage == 1:
            return
        target_Q = self.saved_critic(next_state, next_action)
        target_Q = reward + ((1 - done) * gamma * target_Q).detach()
        adv_error = target_Q - Q
        adv_error_list = []
        for e in adv_error:
            if e < self.sample_from_new:
                adv_error_list.append(abs(e[0].item()))
            else:
                adv_error_list.append(abs(1/e[0].item()))
        if writer is not None:
            writer.add_scalar("adv_error", sum(adv_error_list) / len(adv_error_list), global_step=episode)
        self.priority_update(self.new_buffer, indices, adv_error_list[:self.sample_from_new])
        if self.sample_from_old !=0:
            self.priority_update(self.old_buffer, indices, adv_error_list[self.sample_from_new+1:])
        self.recalculate_lamda()

    def priority_update(self, buffer, indices, priorities):
        for i, p in zip(indices, priorities):
            p = p ** self.alpha
            if p > self.max_p:
                self.max_p = p
            buffer.val_update(i, p)

    def calculate_idx_diff(self,indices):
        avg_sample_index_delta = 0
        for batch_idx, buffer_idx in enumerate(indices):
            if batch_idx < self.sample_from_new:
                avg_sample_index_delta += (buffer_idx + self.old_buffer.size)
            else:
                avg_sample_index_delta += (self.old_buffer.size-buffer_idx) + self.new_buffer.size
        avg_sample_index_delta /= len(indices)

if __name__ == '__main__':
    er = AdvPER(100,10,alpha=1,beta=1,verbose=True)
    for i in range(100):
        er.add(["a","a","a","a","a"])
        print("lamda",er.lamda,"new",er.sample_from_new,"old",er.sample_from_old)
        er.recalculate_lamda()
    er.stage_1_to_2()
    print("=========")
    for i in range(100):
        er.add(["b","b","b","b","b"])
        print("lamda",er.lamda,"new",er.sample_from_new,"old",er.sample_from_old)
        # print(er.sample()[0])
        er.recalculate_lamda()