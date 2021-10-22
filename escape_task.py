import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

import gym
import lolgym.envs
from pylol.lib import actions, features, point
from pylol.lib import point

from absl import flags
FLAGS = flags.FLAGS

_NO_OP = [actions.FUNCTIONS.no_op.id]
_MOVE = [actions.FUNCTIONS.move.id]
_SPELL = [actions.FUNCTIONS.spell.id]

import gym
from gym.spaces import Box, Tuple, Discrete, Dict, MultiDiscrete

from dotenv import dotenv_values
import neptune.new as neptune

from absl import flags
from absl import app

FLAGS = flags.FLAGS
flags.DEFINE_string("config_dirs", "/mnt/c/Users/win8t/Desktop/pylol/config.txt", "Path to file containing GameServer and LoL Client directories")
flags.DEFINE_string("host", "192.168.0.16", "Host IP for GameServer, LoL Client and Redis")
flags.DEFINE_integer("epochs", 50, "Number of episodes to run the experiment for")
flags.DEFINE_float("step_multiplier", 1.0, "Run game server x times faster than real-time")
flags.DEFINE_bool("run_client", False, "Controls whether the game client is run or not")
flags.DEFINE_integer("max_steps", 25, "Maximum number of steps per episode")

# (Optional) Initialises Neptune.ai logging
log = False
config = dotenv_values(".env")
if "NEPTUNE_PROJECT" in config and "NEPTUNE_TOKEN" in config:
    if config["NEPTUNE_PROJECT"] and config["NEPTUNE_TOKEN"]:
        run = neptune.init(
            project=config["NEPTUNE_PROJECT"],
            api_token=config["NEPTUNE_TOKEN"])
        log = True

# Use GPU if available
if (torch.cuda.is_available()):
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    print("Device set to:", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("Device set to: CPU")


class RolloutBuffer(object):
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim.shape[0], 1),
            nn.Tanh(),
            nn.Linear(1, 1),
            nn.Tanh(),
            nn.Linear(1, action_dim.n),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim.shape[0], 1),
            nn.Tanh(),
            nn.Linear(1, 1),
            nn.Tanh(),
            nn.Linear(1, 1)
        )

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        action = action.detach()
        action_logprob = action_logprob.detach()

        if log:
            if action == 0:
                run['action_left_logprob'].log(action_logprob)
            else:
                run['action_right_logprob'].log(action_logprob)

        return action.detach(), action_logprob.detach()

    def eval(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprob, dist_entropy, state_values


class PPO(object):
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()
        
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
    
    def forward(self):
        raise NotImplementedError

    def act(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.item()

    def update(self):
        # Monte Carlo estimate of rewards
        rews = []
        discounted_rew = 0
        for rew, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_rew = 0
            discounted_rew = rew + (self.gamma + discounted_rew)
            rews.insert(0, discounted_rew)

        # Normalizing the rewards
        rews = torch.tensor(rews, dtype=torch.float32).to(device)
        rews = (rews - rews.mean()) / (rews.std() + 1e-7)

        # Convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        old_states = torch.unsqueeze(old_states, 1)
        print('[UPDATE] old_states:', old_states, old_states.shape)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Eval old actions and values
            logprobs, state_values, dist_entropy = self.policy.eval(old_states, old_actions)

            # Match state_values tensor dims with rews tensor
            state_values = torch.squeeze(state_values)

            # Find the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding surrogate loss
            advantages = rews - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # Final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rews) - 0.01 * dist_entropy

            """
            if log:
                run["policy_loss"].log(loss)
            """

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
    
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

def convert_action(raw_obs, act):
    act_x = 8 if act else 0
    act_y = 4
    target_pos = point.Point(raw_obs[0].observation["me_unit"].position_x,
                                raw_obs[0].observation["me_unit"].position_y)
    act = [
        [1, point.Point(act_x, act_y)],
        _NO_OP # _SPELL + [[0], target_pos]
    ]

    return act

def test(host, epochs, max_steps, obs_space, act_space, lr_actor, lr_critic, model_path):
    # Initialize gym environment
    env_name = "LoLGame-v0"
    env = gym.make(env_name)
    env.settings["map_name"] = "Old Summoners Rift"
    env.settings["human_observer"] = FLAGS.run_client # Set to true to run league client
    env.settings["host"] = host # Set this using "hostname -i" ip on Linux
    env.settings["players"] = "Ezreal.BLUE,Ezreal.PURPLE"
    env.settings["config_path"] = FLAGS.config_dirs
    env.settings["step_multiplier"] = FLAGS.step_multiplier

    # Initialize actor-critic agent
    eps_clip = 0.2
    gamma = 0.99
    K_epochs = 1 # Originaly 80
    agent = PPO(obs_space, act_space, lr_actor, lr_critic, gamma, K_epochs, eps_clip)

    # Load pre-trained model
    agent.load(model_path)

    total_steps = 0

    for epoch in range(epochs):
        obs = env.reset()
        
        env.teleport(1, point.Point(7100.0, 7000.0))
        env.teleport(2, point.Point(7500.0, 7000.0))
        
        raw_obs = obs
        obs = np.array(raw_obs[0].observation["enemy_unit"].distance_to_me)[None]

        rews = []
        steps = 0

        while True:
            steps += 1
            total_steps += 1

            # Select action with policy
            act = agent.act(obs[None]) # NOTE: arr[None] wraps arr in another [] 
            act = convert_action(raw_obs, act)
            obs, rew, done, _ = env.step(act)
            raw_obs = obs
            obs = np.array(raw_obs[0].observation["enemy_unit"].distance_to_me)[None]

            # Extract distance from rews
            #rews = [ob[0] / 1000.0 for ob in obs]
            rew = +(raw_obs[0].observation["enemy_unit"].distance_to_me / 1000.0)
            rews.append(rew)

            # Break if episode is over
            if any(done) or steps == max_steps:
                # Announce episode number and rew
                env.broadcast_msg(f"episode No: {epoch}, rew: {sum(rews)}")

                # Break
                break

        rews = []

    # Close environment
    env.close()

def train(host, epochs, max_steps, obs_space, act_space, lr_actor, lr_critic):
    # Initialize gym environment
    env_name = "LoLGame-v0"
    env = gym.make(env_name)
    env.settings["map_name"] = "Old Summoners Rift"
    env.settings["human_observer"] = FLAGS.run_client # Set to true to run league client
    env.settings["host"] = host # Set this using "hostname -i" ip on Linux
    env.settings["players"] = "Ezreal.BLUE,Ezreal.PURPLE"
    env.settings["config_path"] = FLAGS.config_dirs
    env.settings["step_multiplier"] = FLAGS.step_multiplier

    # Initialize actor-critic agent
    eps_clip = 0.2
    gamma = 0.99
    K_epochs = 80 # K_epochs = 1 # Originaly 80
    agent = PPO(obs_space, act_space, lr_actor, lr_critic, gamma, K_epochs, eps_clip)

    run["K_epochs"] = K_epochs

    # Training process
    update_step = max_steps * 1

    # Random seed
    random_seed = 0
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    # Checkpoint
    checkpoint_path = "PPO_{}_{}_.pth".format(env_name, random_seed)

    total_steps = 0

    for epoch in range(FLAGS.epochs):
        steps = 0
        obs = env.reset()
        
        env.teleport(1, point.Point(7100.0, 7000.0))
        env.teleport(2, point.Point(7500.0, 7000.0))
        
        raw_obs = obs
        obs = np.array(raw_obs[0].observation["enemy_unit"].distance_to_me)[None]
        print(f'obs: {epoch}, step: 0')

        rews = []
        steps = 0

        while True:
            steps += 1
            total_steps += 1

            print(f'obs: {epoch}, step: {steps}')

            # Select action with policy
            act = agent.act(obs[None]) # NOTE: arr[None] wraps arr in another [] 
            act = convert_action(raw_obs, act)
            obs, rew, done, _ = env.step(act)
            raw_obs = obs
            obs = np.array(raw_obs[0].observation["enemy_unit"].distance_to_me)[None]

            # Extract distance from rews
            #rews = [ob[0] / 1000.0 for ob in obs]
            rew = +(raw_obs[0].observation["enemy_unit"].distance_to_me / 1000.0)
            rews.append(rew)

            # Saving reward and is_terminals
            agent.buffer.rewards.append(rew)
            agent.buffer.is_terminals.append(done[0])

            # Update PPO agent
            if total_steps % update_step == 0:
                agent.update()
                agent.save(checkpoint_path)

            # Break if episode is over
            if any(done) or steps == max_steps:
                # Announce episode number and rew
                env.broadcast_msg(f"episode No: {epoch}, rew: {sum(rews)}")

                if log:
                    run["reward"].log(sum(rews))

                # Break
                break

        rews = []

    # Close environment
    env.close()

def main(unused_argv):
    obs_space = Box(low=0, high=24000, shape=(1,), dtype=np.float32)
    act_space = Discrete(2)
    
    # lr_actor  = 0.0003
    lr_actor  = 0.003

    #lr_critic = 0.001
    lr_critic = 0.01

    if log:
        run["lr_actor"] = lr_actor
        run["lr_critic"] = lr_critic

    host = FLAGS.host
    epochs = FLAGS.epochs
    max_steps = FLAGS.max_steps

    # Checkpoint
    train(host, epochs, max_steps, obs_space, act_space, lr_actor, lr_critic)
    # checkpoint_path = "PPO_LoLGame-v0_0_200epochs.pth"
    # test(host, epochs, max_steps, obs_space, act_space, lr_actor, lr_critic, checkpoint_path)

def entry_point():
    app.run(main)

if __name__ == "__main__":
    app.run(main)