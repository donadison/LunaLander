import gym
from stable_baselines3 import DQN  # Możesz zmienić na PPO/A2C, jeśli wolisz inne algorytmy
from MoonLanderEnvGym import MoonLanderEnv
import pygame
import sys
import numpy as np


def train_model(env, total_timesteps=10000):
    """
    Funkcja trenuje model DQN na podanym środowisku.
    """
    print("Rozpoczynanie treningu...")
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    print("Trening zakończony!")
    model.save("moon_lander_model")
    return model


def run_model(env):
    """
    Funkcja uruchamia wyuczony model i symuluje działanie środowiska.
    """
    print("Ładowanie modelu...")
    model = DQN.load("moon_lander_model")
    print("Model załadowany. Rozpoczynanie symulacji...")
    
    # Inicjalizacja Pygame
    pygame.init()
    screen = pygame.display.set_mode((600, 400))  # Tworzenie okna gry
    pygame.display.set_caption("Moon Lander")
    clock = pygame.time.Clock()  # Ustawienie zegara
    
    obs = env.reset()  # Reset środowiska
    if isinstance(obs, tuple):  # Obsługa różnych wersji API Gym
        obs = obs[0]

    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        env.render()  # Render środowiska
        action, _states = model.predict(obs, deterministic=True)  # Uzyskaj akcję od modelu
        obs, reward, done, truncated, info = env.step(action)  # Wykonaj akcję

        if isinstance(obs, tuple):  # Obsługa różnych formatów Gym
            obs = obs[0]

        # Opcjonalnie: dodanie opóźnienia w celu płynniejszego działania
        clock.tick(30)  # Ograniczenie do 30 FPS

    print("Symulacja zakończona.")
    pygame.quit()


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
