import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a)) * self.max_action
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)

        q = F.relu(self.l1(state_action))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q


class DDPG(object):
    def __init__(self, lr, state_dim, action_dim, max_action,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, replay_buffer, n_iter, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay):
        for i in range(n_iter):
            # Sample a batch of transitions from replay buffer:
            if replay_buffer.size < batch_size:
                return
            sample = replay_buffer.sample()
            if sample is None:
                return
            state, action_, reward, next_state, done, _, _ = sample
            state = torch.FloatTensor(state).to(self.device)
            action = torch.FloatTensor(action_).to(self.device)
            reward = torch.FloatTensor(reward).reshape((batch_size,1)).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            done = torch.FloatTensor(done).reshape((batch_size,1)).to(self.device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + ((1-done) * gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_((1 - polyak) * param.data + polyak * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_((1 - polyak) * param.data + polyak * target_param.data)

    def save(self, directory, step):
        step = str(step)
        torch.save(self.actor.state_dict(), '%s/%sk_actor.pth' % (directory, step))
        torch.save(self.actor_target.state_dict(), '%s/%sk_actor_target.pth' % (directory, step))

        torch.save(self.critic.state_dict(), '%s/%sk_critic.pth' % (directory, step))
        torch.save(self.critic_target.state_dict(), '%s/%sk_critic_target.pth' % (directory, step))


    def load(self, restore_appendix):
        self.actor.load_state_dict(
            torch.load(f'{restore_appendix}actor.pth', map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(
            torch.load(f'{restore_appendix}actor_target.pth', map_location=lambda storage, loc: storage))

        self.critic.load_state_dict(
            torch.load(f'{restore_appendix}critic.pth', map_location=lambda storage, loc: storage))
        self.critic_target.load_state_dict(
            torch.load(f'{restore_appendix}critic_target.pth', map_location=lambda storage, loc: storage))

    def load_actor(self, directory, name):
        self.actor.load_state_dict(
            torch.load('%s/%sk_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(
            torch.load('%s/%sk_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))



