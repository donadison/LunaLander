import pygame
import numpy as np
from stable_baselines3 import DQN  # Załóżmy, że używasz PPO (możesz zmienić na inny model)
from MoonLanderEnvGym_copy import MoonLanderEnv  # Import środowiska

# Parametry okna
WIDTH, HEIGHT = 600, 400
FPS = 30

# Zapytaj użytkownika o tryb sterowania
mode = input("Wpisz 'run' dla uruchomienia modelu RL lub 'play' dla ręcznego sterowania: ").strip().lower()

if mode == 'run':
    USE_MANUAL_CONTROL = False  # Model RL steruje autonomicznie
elif mode == 'play':
    USE_MANUAL_CONTROL = True  # Użytkownik steruje ręcznie
else:
    print("Nieznany tryb, domyślnie ustawiamy 'run'.")
    USE_MANUAL_CONTROL = False

# Inicjalizacja środowiska
env = MoonLanderEnv()
state, _ = env.reset()

# Jeśli używasz modelu RL, załaduj go
model_path = "dqn_moonlander"  # Ścieżka do wytrenowanego modelu
if not USE_MANUAL_CONTROL:
    model = DQN.load(model_path)  # Ładujemy model PPO (lub inny, zależnie od używanego)

# Uruchomienie PyGame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("MoonLander RL")
clock = pygame.time.Clock()

running = True
total_reward = 0

def get_manual_action():
    """Funkcja, która zwraca akcję na podstawie wejścia użytkownika z klawiatury"""
    keys = pygame.key.get_pressed()  # Pobiera stan wszystkich klawiszy
    if keys[pygame.K_LEFT]:
        return 1  # Obrót w lewo
    elif keys[pygame.K_RIGHT]:
        return 2  # Obrót w prawo
    elif keys[pygame.K_UP]:
        return 3  # Thrust
    return 0  # Brak akcji

try:
    while running:
        # Obsługa zdarzeń PyGame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Jeżeli manualne sterowanie jest włączone, wykonaj akcję manualnie
        if USE_MANUAL_CONTROL:
            action = get_manual_action()
        else:
            # Model podejmuje decyzje
            action, _states = model.predict(state, deterministic=True)  # Predykcja akcji przez model

        # Wykonaj krok w środowisku
        state, reward, done, truncated, info = env.step(action)
        total_reward = reward

        # Renderuj środowisko (można zmienić 'human' na inne tryby w zależności od potrzeb)
        env.render(mode="human")  # Renderuje w oknie PyGame

        # Zatrzymaj, jeśli epizod się skończy
        if done or truncated:
            print(f"Epizod zakończony! Otrzymano nagrodę: {total_reward:.2f}")
            state, _ = env.reset()
            total_reward = 0

        # Kontrola FPS (zapewnia odpowiednią płynność)
        clock.tick(FPS)

except KeyboardInterrupt:
    print("Przerwano ręcznie.")

finally:
    env.close()  # Zamyka środowisko
    pygame.quit()  # Zamyka PyGame
