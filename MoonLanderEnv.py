import pygame
import numpy as np
import random

import Surface
import Background
import Lander
import Plotter

# Parametry ekranu
WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT), flags=pygame.HIDDEN)
pygame.display.set_caption("Moonlander RL")

# Kolory
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
BLACK = (0, 0, 0)

# Parametry lądownika
LANDER_SIZE = (20, 30)
GRAVITY = 0.1
THRUST = -0.2
ROTATION_SPEED = 3  # Stopnie na klatkę
MAX_LANDING_SPEED = 0.5
MAX_LANDING_ANGLE = 3  # Maksymalny dopuszczalny kąt nachylenia w stopniach

# Parametry platformy
PLATFORM_WIDTH = 50
PLATFORM_HEIGHT = 10
PLATFORM_POS = [WIDTH // 2 - PLATFORM_WIDTH // 2, HEIGHT - 30]
# Randomized starting position for the lander
def get_random_lander_start_pos():
    min_x = LANDER_SIZE[0] / 2 + 5  # Add padding on the left
    max_x = WIDTH - LANDER_SIZE[0] / 2 + 5  # Add padding on the right
    x_start = random.uniform(min_x, max_x)
    y_start = random.uniform(HEIGHT // 10, HEIGHT // 2)  # Ensure Y > half of HEIGHT
    return [x_start, y_start]


# Lander starting position
LANDER_START_POS = get_random_lander_start_pos()

# Klasa środowiska
class MoonLanderEnv:
    def __init__(self):
        self.lander = Lander.MoonLander(LANDER_START_POS, LANDER_SIZE)
        self.background = Background.StarryBackground(WIDTH, HEIGHT, num_stars=200)  # 200 gwiazd
        self.moon_surface = Surface.MoonSurface(WIDTH, HEIGHT, max_height=30, roughness=30)
        self.plot = Plotter.Plotter('rewards.csv')
        self.wind = 0
        self.done = False
        self.reward = 0.0
        # Reward configuration (can be tuned)
        self.reward_config = {
            'distance_weight': 1,
            'speed_weight': 1,
            'angle_weight': 0.5
        }
        self.reason = "Brak powodu"
        self.time_limit = 500  # Maximum steps per episode
        self.step_count = 0    # Step counter

    def reset(self):
        """Resetuje środowisko do stanu początkowego."""
        self.lander = Lander.MoonLander(get_random_lander_start_pos(), LANDER_SIZE)
        self.background = Background.StarryBackground(WIDTH, HEIGHT, num_stars=200)  # Regeneruj tło
        self.moon_surface = Surface.MoonSurface(WIDTH, HEIGHT, max_height=30, roughness=30)
        self.done = False
        self.reward = 0.0
        self.reason = "Brak powodu"
        self.step_count = 0
        return self.get_state()

    def step(self, action):
        self.step_count += 1  # Increment step counter
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

        self.wind = random.uniform(-0.2, 0.2)

        # Aktualizuj lądownik
        self.lander.update(thrust, rotation, GRAVITY, self.wind, left_action, right_action)
        self.reward = self.calculate_distance_reward()

        if self.lander.pos[1] < 0:
            self.done = True
            self.reward -= 100
            self.reason = "Kowboj przesadził i poleciał prosto w gwiazdy! Orbitę opuścił szybciej niż pędzący meteoryt."

        if self.lander.pos[0] < 0 or self.lander.pos[0] + self.lander.size[0] > WIDTH:
            self.done = True
            self.reward -= 100
            self.reason = "Dziki zachód wymaga dzikiej precyzji, kowboju!"

        if self.moon_surface.check_collision(self.lander):
            self.done = True
            self.reward -= 100
            self.reason = "Kowboj wpadł na skały jak kaktus w burzę piaskową!"

        # Check if the time limit has been reached
        if self.step_count >= self.time_limit:
            self.done = True
            self.reward -= 50
            self.reason = "Czas się skończył, kowboju! Nie wykonano misji na czas."

        # Sprawdź kolizję z platformą
        if (
                self.lander.pos[1] + self.lander.size[1] >= PLATFORM_POS[1]
                and PLATFORM_POS[0] <= self.lander.pos[0] <= PLATFORM_POS[0] + PLATFORM_WIDTH
        ):
            self.lander.pos[1] = PLATFORM_POS[1] - self.lander.size[1]
            self.done = True
            self._check_landing()

        return self.get_state(), self.reward, self.reason, self.done

    def _check_landing(self):
        """Sprawdza, czy lądowanie było udane."""
        too_fast = abs(self.lander.velocity[1]) > MAX_LANDING_SPEED
        too_tilted = abs(self.lander.angle) > MAX_LANDING_ANGLE

        if not too_fast and not too_tilted:
            # Perfect landing
            self.reward += 100
            self.reason = "Howdy, Houston! Lądownik zaparkowany prosto jak w saloonie."
        elif too_fast and not too_tilted:
            self.reward += 75 + self.calculate_velocity_penalty()
            self.reason = "Zwolnij kowboju! Platforma ledwo ustała!"
        elif too_tilted and not too_fast:
            self.reward += 75 + self.calculate_angle_penalty()
            self.reason = "Krzywo jak dach stodoły po burzy, ale się trzyma!"
        else:
            # Poor landing
            self.reward += 75 + self.calculate_angle_penalty() + self.calculate_velocity_penalty()
            self.reason = "Za szybko i za krzywo – lądowanie jak u pijącego szeryfa w miasteczku!"

    def calculate_distance_reward(self):
        center_x = PLATFORM_POS[0] + PLATFORM_WIDTH / 2
        top_y = PLATFORM_POS[1]
        horizontal_distance = np.abs(self.lander.pos[0] - center_x)  # Horizontal distance to the platform center
        vertical_distance = np.abs(self.lander.pos[1] - top_y)  # Vertical distance to the top of the platform
        euclidean_distance = np.sqrt(horizontal_distance ** 2 + vertical_distance ** 2)  # Euclidean distance
        reward = -euclidean_distance * self.reward_config['distance_weight']  # Negative reward based on distance
        return reward

    def calculate_angle_penalty(self):
        reward = -abs(self.lander.angle) * self.reward_config['angle_weight']
        return reward

    def calculate_velocity_penalty(self):
        reward = -abs(self.lander.velocity[1]) * self.reward_config['speed_weight']
        return reward

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
        wind_text = font.render(f"Wiatr: {self.wind:.2f}", True, WHITE)
        screen.blit(velocity_text, (10, 10))
        screen.blit(angle_text, (10, 40))
        screen.blit(wind_text, (10, 70))

        pygame.display.flip()

    def get_state(self):
        """Zwraca obecny stan środowiska jako wektor."""
        platform_center_x = PLATFORM_POS[0] + PLATFORM_WIDTH / 2
        platform_y = PLATFORM_POS[1]  # Wysokość platformy
        return np.array([
            self.lander.pos[0],  # Pozycja X lądownika
            self.lander.pos[1],  # Pozycja Y lądownika
            self.lander.velocity[0],  # Prędkość X
            self.lander.velocity[1],  # Prędkość Y
            self.lander.angle,  # Kąt lądownika
        ], dtype=np.float32)