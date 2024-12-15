import gym
from stable_baselines3 import DQN  # Możesz zmienić na PPO/A2C, jeśli wolisz inne algorytmy
from MoonLanderEnvGym import MoonLanderEnv
import pygame

pygame.init()
pygame.display.set_mode((600, 400), flags=pygame.SHOWN)

def train_model(env, total_timesteps=10000):
    """
    Funkcja trenuje model DQN nta podanym środowisku.
    """
    print("Rozpoczynanie treningu...")
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    print("Trening zakończony!")
    model.save("moon_lander_model")
    return model

def run_model(env):
    model = DQN.load("moon_lander_model")  # Wczytaj wyuczony model
    obs = env.reset()  # Obsługa resetu w VecEnv
    if isinstance(obs, tuple):  # Dostosowanie do formatu Gymnasium
        obs = obs[0]  # Pobierz tylko `obs`, ignoruj `info`

    for _ in range(1000):  # Liczba kroków w symulacji
        env.render()  # Renderuj środowisko
        action, _states = model.predict(obs, deterministic=True)  # Uzyskaj akcję od modelu
        obs, reward, done, truncated, info = env.step(action)  # Wykonaj akcję
        if isinstance(obs, tuple):  # Dostosowanie w przypadku Gymnasium
            obs = obs[0]
        if done or truncated:
            print(f"Epizod zakończony: {info.get('reason', 'Brak informacji')}")
            obs = env.reset()
            if isinstance(obs, tuple):  # Obsługa resetu w Gymnasium
                obs = obs[0]

if __name__ == "__main__":
    # Inicjalizacja środowiska
    env = MoonLanderEnv()

    # Opcja: trening lub załaduj istniejący model
    mode = input("Wybierz tryb: 'train' (trenowanie) lub 'run' (uruchomienie modelu): ").strip().lower()

    if mode == "train":
        train_model(env, total_timesteps=10000)  # Możesz zmienić liczbę kroków
    elif mode == "run":
        run_model(env)
    else:
        print("Niepoprawny wybór. Użyj 'train' lub 'run'.")
