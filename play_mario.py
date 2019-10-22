import gym
import gym_super_mario_bros
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

env = gym_super_mario_bros.make('SuperMarioBros2-v0')
env = DummyVecEnv([lambda: env])


model = PPO2.load("ppo_mario.p")

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
