
import gym_super_mario_bros
import gym
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv


env = gym_super_mario_bros.make('SuperMarioBros-v0')
#env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = DummyVecEnv([lambda: env])

#model = PPO2(MlpPolicy, env, verbose=1)
model = PPO2.load("ppo_mario32_2mil (4).p")

# Enjoy trained agent
obs = env.reset()
for i in range(200000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
