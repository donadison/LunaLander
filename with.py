import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
import numpy as np

# Import środowiska MoonLanderEnv
from MoonLanderEnvGym_copy import MoonLanderEnv

# Wrapper zgodny z Gymnasium
class GymWrapper(gym.Env):
    def __init__(self):
        super(GymWrapper, self).__init__()
        self.env = MoonLanderEnv()
        self.action_space = spaces.Discrete(4)  # 4 akcje: nic, lewo, prawo, ciąg
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32  # 5 zmiennych stanu
        )

    def step(self, action):
        state, reward, reason, done = self.env.step(action)
        info = {"reason": reason}  # Gymnasium wymaga słownika 'info'
        return np.array(state, dtype=np.float32), reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Dodanie obsługi seeda
        return np.array(self.env.reset(), dtype=np.float32), {}

    def render(self, mode="human"):
        self.env.render()

# Tworzenie środowiska
env = GymWrapper()

# Sprawdzenie środowiska
check_env(env, warn=True)  # Wykrywa problemy ze środowiskiem
print("Środowisko jest zgodne z Gymnasium API!")
