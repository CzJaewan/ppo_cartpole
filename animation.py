from ppo import PPOAgent
import torch
import gym
from matplotlib import animation, rc
import matplotlib
import random as rand

env= gym.make('CartPole-v0')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
agent = PPOAgent(state_size=env.observation_space.shape[0],
                 action_size=env.action_space.n, 
                 seed=0,
                 hidden_layers=[64,64],
                 lr_policy=1e-4, 
                 use_reset=True,
                 device=device)

agent.policy.load_state_dict(torch.load('policy_cartpole.pth', map_location=lambda storage, loc: storage))

rc('animation', html='jshtml')

# function to animate a list of frames

def play(env, policy, time):
    frame1 = env.reset()
    
    for i in range(time):
        
        anim_frames.append(env.render(mode='rgb_array'))
        frame_input = torch.from_numpy(frame1).unsqueeze(0).float().to(device)
        action = policy.act(frame_input)['a'].cpu().numpy()
        frame1, _, is_done, _ = env.step(int(action))

        if is_done:
            print("reward :", i+1)
            break
    
    env.close()
    
    #return animate_frames(anim_frames)

if __name__ == '__main__':
    play(env, agent.policy, 200)
