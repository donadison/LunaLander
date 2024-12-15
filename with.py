import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

# Poprawiona klasa wrappera dla Gymnasium
class MoonLanderGymWrapper(gym.Env):
    def __init__(self, moon_lander_env):
        self.env = moon_lander_env
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.action_space = Discrete(4)  # 0: No action, 1: Left, 2: Right, 3: Thrust

    def reset(self, *, seed=None, options=None):
        state = self.env.reset()
        return state, {}

    def step(self, action):
        # Pobierz wartości ze środowiska
        state, reward, done, info = self.env.step(action)
        
        # Naprawa błędu z `done`
        if not isinstance(done, bool):
            print(f"Debug: wartość `done` = {done}, typ = {type(done)}")
            done = False  # Jeśli `done` nie jest bool, ustaw domyślną wartość `False`

        # Naprawa błędu z `info`
        if not isinstance(info, dict):
            print(f"Debug: wartość `info` = {info}, typ = {type(info)}")
            info = {}  # Jeśli `info` nie jest słownikiem, ustaw domyślną wartość pustego słownika

        # Obsługa gymnasium: `terminated` i `truncated`
        terminated = done
        truncated = False  # Jeśli Twoje środowisko nie ma logiki "truncated"

        return state, reward, terminated, truncated, info

    def render(self):
        self.env.render()


# Main Training Loop
def main():
    from MoonLanderEnv import MoonLanderEnv  # Importuj swoje środowisko MoonLanderEnv
    
    # Tworzenie środowiska
    custom_env = MoonLanderEnv()
    wrapped_env = MoonLanderGymWrapper(custom_env)

    # Sprawdzanie środowiska pod kątem zgodności z Gymnasium
    check_env(wrapped_env, warn=True)

    # Tworzenie modelu DQN
    model = DQN(
        policy="MlpPolicy",  # Standardowa sieć MLP
        env=wrapped_env,     # Nasze środowisko
        learning_rate=1e-3,  # Parametry algorytmu
        buffer_size=10000,
        learning_starts=1000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        verbose=1,
        tensorboard_log="./dqn_moonlander_tensorboard/"
    )

    # Trenowanie modelu
    model.learn(total_timesteps=100)

    # Zapisywanie modelu
    model.save("dqn_moonlander_model")

    # Testowanie modelu
    obs, _ = wrapped_env.reset()
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = wrapped_env.step(action)
        wrapped_env.render()
        if done:
            obs, _ = wrapped_env.reset()

if __name__ == "__main__":
    main()
