import pygame
import numpy as np
import math
import random

# Inicjalizacja Pygame
pygame.init()

# Parametry ekranu
WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Moonlander RL")

# Kolory
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Parametry lądownika
LANDER_SIZE = (20, 40)
GRAVITY = 0.1
THRUST = -0.2
ROTATION_SPEED = 5  # Stopnie na klatkę
LANDER_START_POS = [WIDTH // 2, 50]
MAX_LANDING_SPEED = 2
MAX_LANDING_ANGLE = 15  # Maksymalny dopuszczalny kąt nachylenia w stopniach

# Parametry platformy
PLATFORM_WIDTH = 100
PLATFORM_HEIGHT = 10
PLATFORM_POS = [WIDTH // 2 - PLATFORM_WIDTH // 2, HEIGHT - 30]

# Klasa środowiska
class MoonLanderEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        """Resetuje środowisko do stanu początkowego."""
        self.lander_pos = np.array(LANDER_START_POS, dtype=np.float32)
        self.lander_velocity = np.array([0, 0], dtype=np.float32)
        self.lander_angle = 0  # Kąt nachylenia statku w stopniach
        self.wind = 0  # Początkowy wiatr
        self.done = False
        self.reward = 0
        return self.get_state()

    def step(self, action):
        """
        Wykonuje akcję i aktualizuje środowisko.
        action: 0 (nic), 1 (obrót w lewo), 2 (obrót w prawo), 3 (thrust w górę).
        """
        # Obrót lądownika
        if action == 1:  # Obrót w lewo
            self.lander_angle += ROTATION_SPEED
        elif action == 2:  # Obrót w prawo
            self.lander_angle -= ROTATION_SPEED

        # Silnik
        if action == 3:  # Thrust w kierunku wskazanym przez kąt
            rad_angle = math.radians(self.lander_angle)
            thrust_x = math.sin(rad_angle) * THRUST
            thrust_y = math.cos(rad_angle) * THRUST
            self.lander_velocity[0] += thrust_x
            self.lander_velocity[1] += thrust_y

        # Dodanie grawitacji
        self.lander_velocity[1] += GRAVITY

        # Dodanie wiatru
        self.wind = random.uniform(-0.20, 0.20)  # Losowy wiatr w zakresie
        self.lander_velocity[0] += self.wind

        # Aktualizacja pozycji
        self.lander_pos += self.lander_velocity

        # Sprawdzenie kolizji z ziemią
        if self.lander_pos[1] + LANDER_SIZE[1] >= HEIGHT:
            self.lander_pos[1] = HEIGHT - LANDER_SIZE[1]
            self.done = True
            self._check_landing()

        # Sprawdzenie kolizji ze ścianami
        if self.lander_pos[0] < 0 or self.lander_pos[0] + LANDER_SIZE[0] > WIDTH:
            self.done = True
            self.reward = -100  # Kolizja ze ścianą

        # Sprawdzenie kolizji z platformą
        if self.lander_pos[1] + LANDER_SIZE[1] >= PLATFORM_POS[1] and PLATFORM_POS[0] <= self.lander_pos[0] <= PLATFORM_POS[0] + PLATFORM_WIDTH:
            self.lander_pos[1] = PLATFORM_POS[1] - LANDER_SIZE[1]  # Ustalamy lądownik na platformie
            self.done = True
            self._check_landing()

        return self.get_state(), self.reward, self.done, {}

    def render(self):
        """Rysuje aktualny stan środowiska."""
        screen.fill(WHITE)

        # Rysowanie platformy
        pygame.draw.rect(screen, GREEN, (*PLATFORM_POS, PLATFORM_WIDTH, PLATFORM_HEIGHT))

        # Rysowanie lądownika (z rotacją)
        rotated_lander = pygame.Surface(LANDER_SIZE, pygame.SRCALPHA)
        rotated_lander.fill(RED)
        rotated_lander = pygame.transform.rotate(rotated_lander, self.lander_angle)
        rect = rotated_lander.get_rect(center=(self.lander_pos[0] + LANDER_SIZE[0] / 2, self.lander_pos[1] + LANDER_SIZE[1] / 2))
        screen.blit(rotated_lander, rect.topleft)

        # Wyświetlanie prędkości, kąta i wiatru
        font = pygame.font.SysFont(None, 24)
        velocity_text = font.render(f"Prędkość X: {self.lander_velocity[0]:.2f}  Prędkość Y: {self.lander_velocity[1]:.2f}", True, BLACK)
        angle_text = font.render(f"Kąt: {self.lander_angle:.2f}", True, BLACK)
        wind_text = font.render(f"Wiatr: {self.wind:.2f}", True, BLACK)
        screen.blit(velocity_text, (10, 10))
        screen.blit(angle_text, (10, 40))
        screen.blit(wind_text, (10, 70))

        pygame.display.flip()

    def get_state(self):
        """Zwraca obecny stan środowiska jako wektor."""
        return np.array([
            self.lander_pos[0], self.lander_pos[1],
            self.lander_velocity[0], self.lander_velocity[1],
            self.lander_angle
        ], dtype=np.float32)

    def _check_landing(self):
        """Sprawdza, czy lądowanie było poprawne."""
        if PLATFORM_POS[0] <= self.lander_pos[0] <= PLATFORM_POS[0] + PLATFORM_WIDTH:
            if abs(self.lander_velocity[1]) <= MAX_LANDING_SPEED and abs(self.lander_angle) <= MAX_LANDING_ANGLE:
                self.reward = 100  # Udane lądowanie
            else:
                self.reward = -100  # Zbyt szybkie lub krzywe lądowanie
        else:
            self.reward = -100  # Lądowanie poza platformą

# Główna pętla gry
def main():
    env = MoonLanderEnv()
    clock = pygame.time.Clock()
    state = env.reset()

    running = True
    while running:
        # Obsługa zdarzeń
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Wybór akcji
        keys = pygame.key.get_pressed()
        action = 0
        if keys[pygame.K_LEFT]:
            action = 1
        elif keys[pygame.K_RIGHT]:
            action = 2
        elif keys[pygame.K_UP]:
            action = 3

        # Wykonanie akcji
        state, reward, done, _ = env.step(action)

        # Renderowanie
        env.render()

        # Zakończenie epizodu
        if done:
            print(f"Epizod zakończony! Nagroda: {reward}")
            state = env.reset()

        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
