from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import gym
from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from google.colab import files
import pickle
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.vec_env import SubprocVecEnv

# Create and wrap the environment


env = gym_super_mario_bros.make('SuperMarioBros2-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env = DummyVecEnv([lambda: env])

#model = PPO2.load("ppo_mario")
model = PPO2(CnnPolicy, env, verbose=1)
# Train the agent
model.learn(total_timesteps=1000000)

model.save("ppo_mario.p")
#pickle.dump(model, open("mariomodel.sav", 'wb'))
#del model nnb
