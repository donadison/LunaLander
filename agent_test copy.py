import numpy as np
import gymnasium as gym
from MoonLanderEnvGym_copy import MoonLanderEnv  # Zaimportuj klasę środowiska

# Ustawienia testowe
EPISODES = 10  # Liczba epizodów do przetestowania
MAX_STEPS = 500  # Maksymalna liczba kroków na epizod

# Utworzenie środowiska
env = MoonLanderEnv()

# Testowy agent wykonujący losowe akcje
for episode in range(EPISODES):
    print(f"Epizod {episode + 1}/{EPISODES}")
    state, _ = env.reset()  # Reset środowiska
    total_reward = 0  # Całkowita nagroda dla epizodu
    
    for step in range(MAX_STEPS):
        env.render()  # Renderowanie środowiska (opcjonalne, przyspiesza wyłączenie renderowania)
        
        action = env.action_space.sample()  # Losowa akcja
        state, reward, terminated, truncated, info = env.step(action)  # Wykonanie akcji w środowisku
        
        total_reward += reward  # Sumowanie nagród
        
        if terminated or truncated:  # Sprawdzenie, czy epizod zakończony
            print(f"Epizod zakończony na kroku {step + 1} z powodu: {info['reason']}.")
            break
    
    print(f"Całkowita nagroda za epizod: {total_reward:.2f}")

env.close()  # Zamknięcie środowiska