import torch
import torch.optim as optim
import numpy as np
from policy import PolicyNet, ValueNet
from env_client import BlenderEnvClient
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STATE_DIM = 7   # must match blender state vector length
ACTION_DIM = 3  # steer, throttle, brake

# hyperparams (tune)
LR = 3e-4
GAMMA = 0.99
LAMBDA = 0.95
EPS_CLIP = 0.2
EPOCHS = 4
BATCH_SIZE = 64
TRAJECTORY_LEN = 256

def transform_action(raw_action):
    """Transform raw network output to valid action space"""
    steer = np.tanh(raw_action[0])  # [-1, 1]
    throttle = 1.0 / (1.0 + np.exp(-raw_action[1]))  # [0, 1]
    brake = 1.0 / (1.0 + np.exp(-raw_action[2]))  # [0, 1]
    return [steer, throttle, brake]

def select_action(policy, state):
    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    mu, std = policy(state_t)
    dist = torch.distributions.Normal(mu, std)
    action = dist.sample()
    logp = dist.log_prob(action).sum(-1)
    return action.squeeze(0).cpu().numpy(), logp.item()

def compute_gae(rewards, masks, values, gamma=GAMMA, lam=LAMBDA):
    values = np.append(values, 0)
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
    advs = np.array(returns) - values[:-1]
    return np.array(returns), (advs - advs.mean()) / (advs.std() + 1e-8)

def train():
    env = BlenderEnvClient()
    policy = PolicyNet(STATE_DIM, ACTION_DIM).to(device)
    value = ValueNet(STATE_DIM).to(device)
    opt_policy = optim.Adam(policy.parameters(), lr=LR)
    opt_value = optim.Adam(value.parameters(), lr=LR)

    episode = 0
    while True:
        # collect trajectory
        states = []
        actions = []
        logps = []
        rewards = []
        masks = []
        values = []

        for t in range(TRAJECTORY_LEN):
            state = env.recv_state()    # blocking: waits for Blender
            states.append(state)
            st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            val = value(st).item()
            values.append(val)
            
            raw_action, logp = select_action(policy, state)
            action = transform_action(raw_action)
            
            env.send_action(action)
            actions.append(raw_action)  # Store raw for training
            logps.append(logp)
            
            # Better reward: encourage reaching goal and efficient driving
            dist_to_goal = state[-1]
            speed = np.sqrt(state[4]**2 + state[5]**2)  # x,y velocity
            reward = -dist_to_goal * 0.1 + speed * 0.01 - 0.001  # small survival bonus
            
            rewards.append(reward)
            masks.append(1.0) 

        last_state = torch.tensor(states[-1], dtype=torch.float32, device=device).unsqueeze(0)
        last_val = value(last_state).item()
        values.append(last_val)

        returns, advs = compute_gae(rewards, masks, np.array(values))
        
        states_t = torch.tensor(states, dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.float32, device=device)
        old_logps_t = torch.tensor(logps, dtype=torch.float32, device=device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        advs_t = torch.tensor(advs, dtype=torch.float32, device=device)

        # policy / value update (PPO style)
        for _ in range(EPOCHS):
            # simple full-batch update (better to minibatch)
            mu, std = policy(states_t)
            dist = torch.distributions.Normal(mu, std)
            new_logps = dist.log_prob(actions_t).sum(-1)
            ratio = torch.exp(new_logps - old_logps_t)
            surr1 = ratio * advs_t
            surr2 = torch.clamp(ratio, 1.0 - EPS_CLIP, 1.0 + EPS_CLIP) * advs_t
            policy_loss = -torch.min(surr1, surr2).mean()

            value_preds = value(states_t).squeeze()
            value_loss = (returns_t - value_preds).pow(2).mean()

            opt_policy.zero_grad()
            policy_loss.backward()
            opt_policy.step()

            opt_value.zero_grad()
            value_loss.backward()
            opt_value.step()

        episode += 1
        print(f"Episode {episode}, Mean return: {returns_t.mean().item():.3f}, "
              f"Max reward: {max(rewards):.3f}, Min distance: {min([s[-1] for s in states]):.3f}")
        
        env.send_reset()
        time.sleep(0.5)

if __name__ == "__main__":
    train()