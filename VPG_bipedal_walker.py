import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

def compute_returns(rewards, gamma=0.99):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

class PolicyNet(nn.Module):
    def __init__(self, nS, nH, nA): 
        super(PolicyNet, self).__init__()
        self.h = nn.Linear(nS, nH)
        self.h2 = nn.Linear(nH,nH**2)
        self.out_mean = nn.Linear(nH**2, nA)
        self.out_std = nn.Linear(nH**2, nA)

    def forward(self, x):
        x = F.relu(self.h(x))
        x = F.relu(self.h2(x))
        mean = torch.tanh(self.out_mean(x))
        std = F.softplus(self.out_std(x))
        return mean, std

env = gym.make("BipedalWalker-v3", render_mode='human')


learning_rate = 1e-6
gamma = 0.98
MAX_STEPS_PER_EPISODE = 1000

state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
policy = PolicyNet(state_size, 128, action_size)
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
total_rewards = []

for episode in range(5000):
    state = env.reset()
    log_probs = []
    rewards = []
    
    
    for step in range(MAX_STEPS_PER_EPISODE):
        env.render()
        
        actual_state = state[0] if isinstance(state, tuple) else state
        state_tensor = torch.FloatTensor(actual_state).unsqueeze(0)
        mean, std = policy(state_tensor)
        action_distribution = torch.distributions.Normal(mean, std)
        action = action_distribution.sample().clamp(-1, 1).numpy()
        log_prob = action_distribution.log_prob(torch.FloatTensor(action))

        next_state, reward, done, info,_ = env.step(action[0])   # Remove the additional unpacked value 

        log_probs.append(log_prob)
        rewards.append(reward)

        if done:
            break
        
        state = next_state
    
    returns = compute_returns(rewards, gamma)

    policy_loss = []
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append(-log_prob * R)
    policy_loss = torch.cat(policy_loss).sum()
    
    optimizer.zero_grad()
    policy_loss.backward(retain_graph=True)
    optimizer.step()

    print(f"Episode {episode + 1}, Total Reward: {sum(rewards)}")
    total_rewards.append(sum(rewards))

env.close()