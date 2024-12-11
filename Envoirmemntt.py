import pygame
import numpy as np
import random

import Surface
import Background
import Lander

# Inicjalizacja Pygame
pygame.init()

# Parametry ekranu
WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Moonlander RL")

# Kolory
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
ORANGE = (255, 69, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 100)

# Parametry lądownika
LANDER_SIZE = (20, 30)
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
        self.lander = Lander.MoonLander(LANDER_START_POS, LANDER_SIZE)
        self.background = Background.StarryBackground(WIDTH, HEIGHT, num_stars=200)  # 200 gwiazd
        self.moon_surface = Surface.MoonSurface(WIDTH, HEIGHT, max_height=30, roughness=30)
        self.lander_velocity = np.array([0, 0], dtype=np.float32)
        self.lander_angle = 0
        self.wind = 0
        self.done = False
        self.reward = 0
        self.reason = "Brak powodu"

    def reset(self):
        """Resetuje środowisko do stanu początkowego."""
        self.lander = Lander.MoonLander(LANDER_START_POS, LANDER_SIZE)
        self.background = Background.StarryBackground(WIDTH, HEIGHT, num_stars=200)  # Regeneruj tło
        self.moon_surface = Surface.MoonSurface(WIDTH, HEIGHT, max_height=30, roughness=30)
        self.lander_velocity = np.array([0, 0], dtype=np.float32)
        self.lander_angle = 0  # Kąt nachylenia statku w stopniach
        self.wind = 0  # Początkowy wiatr
        self.done = False
        self.reward = 0
        self.reason = "Brak powodu"
        return self.get_state()

    def step(self, action):
        rotation = 0
        thrust = 0
        left_action = False
        right_action = False

        if action == 1:  # Obrót w lewo
            rotation = -ROTATION_SPEED
            left_action = True
        elif action == 2:  # Obrót w prawo
            rotation = ROTATION_SPEED
            right_action = True
        elif action == 3:  # Ciąg
            thrust = THRUST

        # Aktualizuj lądownik
        self.lander.update(thrust, rotation, GRAVITY, random.uniform(-0.20, 0.20), left_action, right_action)

        # Aktualizuj zmienne stanu lądownika
        self.lander_velocity = self.lander.velocity
        self.lander_angle = self.lander.angle

        if self.lander.pos[1] < 0:
            self.done = True
            self.reward = -100  # Kolizja z sufitem
            self.reason = "Kowboj przesadził i poleciał prosto w gwiazdy! Orbitę opuścił szybciej niż pędzący meteoryt."

        if self.lander.pos[0] < 0 or self.lander.pos[0] + self.lander.size[0] > WIDTH:
            self.done = True
            self.reward = -100  # Kolizja ze ścianą
            self.reason = "Dziki zachód wymaga dzikiej precyzji, kowboju!"

        if self.moon_surface.check_collision(self.lander):
            self.done = True
            self.reward = -100  # Zderzenie z powierzchnią księżyca
            self.reason = "Kowboj wpadł na skały jak kaktus w burzę piaskową!"

        # Sprawdź kolizję z platformą
        if (
                self.lander.pos[1] + self.lander.size[1] >= PLATFORM_POS[1]
                and PLATFORM_POS[0] <= self.lander.pos[0] <= PLATFORM_POS[0] + PLATFORM_WIDTH
        ):
            self.lander.pos[1] = PLATFORM_POS[1] - self.lander.size[1]
            self.done = True
            self._check_landing()

        return self.get_state(), self.reward, self.reason, self.done, {}

    def render(self):
        # Rysowanie tła kosmicznego
        screen.fill(BLACK)  # Czarne tło dla kosmosu
        self.background.render(screen)
        self.background.update()  # Aktualizuj gwiazdy dla efektu migotania
        self.moon_surface.render(screen)

        # Rysowanie platformy
        pygame.draw.rect(screen, GRAY, (*PLATFORM_POS, PLATFORM_WIDTH, PLATFORM_HEIGHT))

        support_height = HEIGHT - PLATFORM_POS[
            1]
        support_width = 10
        center_x = PLATFORM_POS[0] + PLATFORM_WIDTH / 2
        pygame.draw.rect(screen, DARK_GRAY, (
            center_x - support_width / 2, PLATFORM_POS[1] + PLATFORM_HEIGHT, support_width, support_height))

        # Renderowanie lądownika
        self.lander.render(screen)

        # Wyświetlanie prędkości, kąta, wiatru
        font = pygame.font.SysFont(None, 24)
        velocity_text = font.render(
            f"Prędkość X: {self.lander.velocity[0]:.2f}  Prędkość Y: {self.lander.velocity[1]:.2f}", True, WHITE)
        angle_text = font.render(f"Kąt: {self.lander.angle:.2f}", True, WHITE)
        wind_text = font.render(f"Wiatr: {random.uniform(-0.20, 0.20):.2f}", True, WHITE)
        screen.blit(velocity_text, (10, 10))
        screen.blit(angle_text, (10, 40))
        screen.blit(wind_text, (10, 70))

        pygame.display.flip()

    def get_state(self):
        """Zwraca obecny stan środowiska jako wektor."""
        return np.array([
            self.lander.pos[0], self.lander.pos[1],
            self.lander_velocity[0], self.lander_velocity[1],
            self.lander_angle
        ], dtype=np.float32)

    def _check_landing(self):
        """Sprawdza, czy lądowanie było udane."""
        if PLATFORM_POS[0] <= self.lander.pos[0] <= PLATFORM_POS[0] + PLATFORM_WIDTH:
            too_fast = abs(self.lander_velocity[1]) > MAX_LANDING_SPEED
            too_tilted = abs(self.lander_angle) > MAX_LANDING_ANGLE

            if not too_fast and not too_tilted:
                self.reward = 100  # Udane lądowanie
                self.reason = "Howdy, Houston! Lądownik zaparkowany prosto jak w saloonie."
            elif too_fast and not too_tilted:
                self.reward = -100  # Za szybkie lądowanie
                self.reason = "Za szybko! Kowboj przestrzelił, a platforma ledwo ustała!"
            elif not too_fast and too_tilted:
                self.reward = -100  # Zły kąt lądowania
                self.reason = "Kąt był bardziej dziki niż mustang na prerii!"
            else:  # Naruszone oba warunki
                self.reward = -100
                self.reason = "Za szybko i za krzywo – lądowanie jak u pijącego szeryfa w miasteczku!"
        else:
            self.reward = -100  # Lądowanie poza platformą
            self.reason = "Kowboj zgubił platformę! To nie dziki zachód, ląduj celniej!"


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
        state, reward, reason, done, _ = env.step(action)

        # Renderowanie
        env.render()

        # Zakończenie epizodu
        if done:
            print(f"Epizod zakończony! {'Nagroda' if reward > 0 else 'Kara'} {reward}, Powód: {reason}")
            state = env.reset()

        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
