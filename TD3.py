import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import replays
from replays.proportional_PER.proportional import ProportionalPER
from replays.adv_replay import AdvPER

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 200)
        self.l2 = nn.Linear(200, 100)
        self.l3 = nn.Linear(100, action_dim)

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

class TD3:
    def __init__(self, lr, state_dim, action_dim, max_action,device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),writer=None):
        self.device = device
        self.writer=writer
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)

        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)

        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, replay_buffer, n_iter, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay, episode):
        td_writer_data = 0
        idx_writer_data = 0
        for iter in range(n_iter):
            # Sample a batch of transitions from replay buffer:
            if replay_buffer.size < batch_size:
                return
            sample = replay_buffer.sample()
            if sample is None:
                return
            state, action_, reward, next_state, done, weights, indices = sample
            state = torch.FloatTensor(state).to(self.device)
            action = torch.FloatTensor(action_).to(self.device)
            reward = torch.FloatTensor(reward).reshape((batch_size,1)).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            done = torch.FloatTensor(done).reshape((batch_size,1)).to(self.device)

            # Select next action according to target policy:
            noise = torch.FloatTensor(action_).data.normal_(0, policy_noise).to(self.device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1-done) * gamma * target_Q).detach()

            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()

            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()

            td_error = target_Q - torch.min(current_Q1, current_Q2)
            td_error_list = []
            for t in td_error:
                td_error_list.append(abs(t[0].item()))
            td_writer_data += sum(td_error_list) / len(td_error_list)

            # TD error
            if isinstance(replay_buffer,replays.adv_replay.AdvPER):
                Q = torch.min(current_Q1, current_Q2)
                replay_buffer.get_q(indices, Q,next_state, next_action,reward,done,gamma,self.writer,episode)
            elif isinstance(replay_buffer,replays.proportional_PER.proportional.ProportionalPER):
                replay_buffer.priority_update(indices, td_error_list)

            # sample_index_delta
            if isinstance(replay_buffer,replays.adv_replay.AdvPER):
                avg_sample_index_delta = replay_buffer.calculate_idx_diff()
            else:
                avg_sample_index_delta = 0
                for i in indices:
                    if i > replay_buffer.get_cursor_idx():
                        avg_sample_index_delta += replay_buffer.max_size - (i-replay_buffer.get_cursor_idx())
                    else:
                        avg_sample_index_delta +=(replay_buffer.get_cursor_idx()-i)
                avg_sample_index_delta /= len(indices)
            idx_writer_data += avg_sample_index_delta

            # Delayed policy updates:
            if i % policy_delay == 0:
                # Compute actor loss:
                actor_loss = -self.critic_1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Polyak averaging update:
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_((polyak * target_param.data) + ((1 - polyak) * param.data))

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_((polyak * target_param.data) + ((1 - polyak) * param.data))

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_((polyak * target_param.data) + ((1 - polyak) * param.data))
        if self.writer is not None:
            self.writer.add_scalar("td_error", td_writer_data/n_iter, global_step=episode)
            self.writer.add_scalar("avg_sample_index_delta", idx_writer_data/n_iter, global_step=episode)

    def save(self, directory, step):
        step = str(step)
        torch.save(self.actor.state_dict(), '%s/%sk_actor.pth' % (directory, step))
        torch.save(self.actor_target.state_dict(), '%s/%sk_actor_target.pth' % (directory, step))

        torch.save(self.critic_1.state_dict(), '%s/%sk_crtic_1.pth' % (directory, step))
        torch.save(self.critic_1_target.state_dict(), '%s/%sk_critic_1_target.pth' % (directory, step))

        torch.save(self.critic_2.state_dict(), '%s/%sk_crtic_2.pth' % (directory, step))
        torch.save(self.critic_2_target.state_dict(), '%s/%sk_critic_2_target.pth' % (directory, step))

    def load(self, restore_appendix):
        self.actor.load_state_dict(torch.load(f'{restore_appendix}actor.pth', map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(torch.load(f'{restore_appendix}actor_target.pth', map_location=lambda storage, loc: storage))

        self.critic_1.load_state_dict(torch.load(f'{restore_appendix}crtic_1.pth', map_location=lambda storage, loc: storage))
        self.critic_1_target.load_state_dict(torch.load(f'{restore_appendix}critic_1_target.pth', map_location=lambda storage, loc: storage))

        self.critic_2.load_state_dict(torch.load(f'{restore_appendix}crtic_2.pth', map_location=lambda storage, loc: storage))
        self.critic_2_target.load_state_dict(torch.load(f'{restore_appendix}critic_2_target.pth', map_location=lambda storage, loc: storage))


    def load_actor(self, directory, name):
        self.actor.load_state_dict(torch.load('%s/%sk_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(torch.load('%s/%sk_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))





