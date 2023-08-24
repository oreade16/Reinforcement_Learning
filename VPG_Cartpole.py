import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Function to compute discounted returns
def compute_returns(rewards, gamma=0.99):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

# Neural network policy
class PolicyNet(nn.Module):
    def __init__(self, nS, nH, nA):
        super(PolicyNet, self).__init__()
        self.fc = nn.Linear(nS, nH)
        self.out = nn.Linear(nH, nA)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return torch.softmax(self.out(x), dim=-1)

env = gym.make("CartPole-v1",render_mode='human')

learning_rate = 1e-2
gamma = 0.99

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
policy = PolicyNet(state_size, 128, action_size)
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

num_episodes = 500
for episode in range(num_episodes):
    state = env.reset()
    log_probs = []
    rewards = []

    while True:
        env.render()
        actual_state = state[0] if isinstance(state, tuple) else state
        state_tensor = torch.FloatTensor(actual_state).unsqueeze(0)
        action_probs = policy(state_tensor)
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample().item()

        next_state, reward, done,info, _ = env.step(action)
        log_probs.append(action_distribution.log_prob(torch.tensor(action)))
        rewards.append(reward)
        
        if done:
            break
        state = next_state

    returns = compute_returns(rewards, gamma)
    policy_loss = []
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append(-log_prob * R)

    policy_loss = torch.stack(policy_loss).sum()

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    print(f"Episode {episode + 1}, Total Reward: {sum(rewards)}")

env.close()
