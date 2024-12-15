import gym
from gym import spaces
import numpy as np
import random
import pygame

# Import Twoich modułów
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
ROTATION_SPEED = 3
MAX_LANDING_SPEED = 0.5
MAX_LANDING_ANGLE = 3

# Parametry platformy
PLATFORM_WIDTH = 50
PLATFORM_HEIGHT = 10
PLATFORM_POS = [WIDTH // 2 - PLATFORM_WIDTH // 2, HEIGHT - 30]

# Funkcja losowania pozycji początkowej lądownika
def get_random_lander_start_pos():
    min_x = LANDER_SIZE[0] / 2 + 5
    max_x = WIDTH - LANDER_SIZE[0] / 2 - 5
    x_start = random.uniform(min_x, max_x)
    y_start = random.uniform(HEIGHT // 10, HEIGHT // 2)
    return [x_start, y_start]

LANDER_START_POS = get_random_lander_start_pos()

# Klasa środowiska zgodna z Gymnasium
class MoonLanderEnv(gym.Env):
    def __init__(self):
        super(MoonLanderEnv, self).__init__()
        # Przestrzeń akcji: 0 = brak akcji, 1 = obrót w lewo, 2 = obrót w prawo, 3 = ciąg
        self.action_space = spaces.Discrete(4)
        # Przestrzeń obserwacji: [x, y, vx, vy, kąt]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -np.inf, -np.inf, -180]),
            high=np.array([WIDTH, HEIGHT, np.inf, np.inf, 180]),
            dtype=np.float32
        )
        # Zmienna stanu
        self.lander = Lander.MoonLander(LANDER_START_POS, LANDER_SIZE)
        self.background = Background.StarryBackground(WIDTH, HEIGHT, num_stars=200)  # 200 gwiazd
        self.moon_surface = Surface.MoonSurface(WIDTH, HEIGHT, max_height=30, roughness=30)
        self.plot = Plotter.Plotter('rewards.csv')
        self.done = False
        self.reward = 0.0
        self.step_count = 0
        self.reason = "Brak powodu"
        self.wind = 0

        # Parametry nagród
        self.reward_config = {
            'distance_weight': 1,
            'speed_weight': 1,
            'angle_weight': 0.5
        }
        self.time_limit = 500
        

        #self.reset()

    def reset(self, seed=None, options=None):
        """Resetuje środowisko do stanu początkowego."""
        super().reset(seed=seed)
        self.lander = Lander.MoonLander(get_random_lander_start_pos(), LANDER_SIZE)
        self.background = Background.StarryBackground(WIDTH, HEIGHT, num_stars=200)
        self.moon_surface = Surface.MoonSurface(WIDTH, HEIGHT, max_height=30, roughness=30)
        self.done = False
        self.reward = 0.0
        self.step_count = 0
        self.reason = "Brak powodu"
        return self.get_state(), {}


    def step(self, action):
        """Wykonuje krok symulacji na podstawie akcji."""
        self.step_count += 1
        rotation = 0
        thrust = 0
        left_action = False
        right_action = False

        if action == 1:  # Obrót w lewo
            rotation = -ROTATION_SPEED
        elif action == 2:  # Obrót w prawo
            rotation = ROTATION_SPEED
        elif action == 3:  # Ciąg
            thrust = THRUST

        self.wind = random.uniform(-0.2, 0.2)
        # Aktualizuj stan lądownika
        self.lander.update(thrust, rotation, GRAVITY, self.wind)

        # Nagroda za odległość od platformy
        self.reward = self.calculate_distance_reward()

        # Warunki zakończenia



        if self.lander.pos[1] < 0 or self.lander.pos[0] < 0 or self.lander.pos[0] + LANDER_SIZE[0] > WIDTH:
            self.done = True
            self.reward -= 100
            self.reason = "Lądownik opuścił pole gry bądź wyleciał poza przestrzeń!"

        if self.moon_surface.check_collision(self.lander):
            self.done = True
            self.reward -= 100
            self.reason = "Kolizja z powierzchnią księżyca!"

        if self.step_count >= self.time_limit:
            self.done = True
            self.reward -= 50
            self.reason = "Przekroczono limit czasu!"

        # Sprawdź lądowanie na platformie
        if (
            self.lander.pos[1] + LANDER_SIZE[1] >= PLATFORM_POS[1]
            and PLATFORM_POS[0] <= self.lander.pos[0] <= PLATFORM_POS[0] + PLATFORM_WIDTH
        ):
            self.lander.pos[1] = PLATFORM_POS[1] - self.lander.size[1]
            self.done = True
            self._check_landing()

        return self.get_state(), self.reward, self.done, False, {"reason": self.reason}

    def render(self):
        """Renderuje środowisko."""
        screen.fill(BLACK)
        self.background.render(screen)
        self.background.update()
        self.moon_surface.render(screen)

        # Rysuj platformę
        pygame.draw.rect(screen, GRAY, (*PLATFORM_POS, PLATFORM_WIDTH, PLATFORM_HEIGHT))
        support_height = HEIGHT - PLATFORM_POS[1]
        support_width = 10
        center_x = PLATFORM_POS[0] + PLATFORM_WIDTH / 2
        pygame.draw.rect(screen, DARK_GRAY, (
            center_x - support_width / 2, PLATFORM_POS[1] + PLATFORM_HEIGHT, support_width, support_height))

        # Renderuj lądownik
        self.lander.render(screen)

        # Wyświetl prędkość, kąt i wiatr
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
        """Zwraca obecny stan środowiska jako wektor NumPy."""
        platform_center_x = PLATFORM_POS[0] + PLATFORM_WIDTH / 2
        platform_y = PLATFORM_POS[1]  # Wysokość platformy
        return np.array([
            self.lander.pos[0],
            self.lander.pos[1],
            self.lander.velocity[0],
            self.lander.velocity[1],
            self.lander.angle,
        ], dtype=np.float32)

    def calculate_distance_reward(self):
        """Oblicza nagrodę na podstawie odległości od platformy."""
        center_x = PLATFORM_POS[0] + PLATFORM_WIDTH / 2
        top_y = PLATFORM_POS[1]
        horizontal_distance = np.abs(self.lander.pos[0] - center_x)
        vertical_distance = np.abs(self.lander.pos[1] - top_y)
        euclidean_distance = np.sqrt(horizontal_distance ** 2 + vertical_distance ** 2)
        return -euclidean_distance * self.reward_config['distance_weight']

    def _check_landing(self):
        """Sprawdza, czy lądowanie było udane."""
        too_fast = abs(self.lander.velocity[1]) > MAX_LANDING_SPEED
        too_tilted = abs(self.lander.angle) > MAX_LANDING_ANGLE

        if not too_fast and not too_tilted:
            self.reward += 100
            self.reason = "Lądowanie udane!"
        elif too_fast or too_tilted:
            self.reward -= 50
            self.reason = "Nieudane lądowanie."

    def calculate_angle_penalty(self):
        reward = -abs(self.lander.angle) * self.reward_config['angle_weight']
        return self.reward

    def calculate_velocity_penalty(self):
        reward = -abs(self.lander.velocity[1]) * self.reward_config['speed_weight']
        return self.reward
