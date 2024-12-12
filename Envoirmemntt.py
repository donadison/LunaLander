import csv
import sys
import os
import pygame
import numpy as np
import random
import torch

import Surface
import Background
import Lander
import DQNAgent

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
ROTATION_SPEED = 1  # Stopnie na klatkę
MAX_LANDING_SPEED = 2
MAX_LANDING_ANGLE = 10  # Maksymalny dopuszczalny kąt nachylenia w stopniach

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
        self.lander_velocity = np.array([0, 0], dtype=np.float32)
        self.lander_angle = 0
        self.wind = 0
        self.done = False
        self.reward = 0.0
        # Reward configuration (can be tuned)
        self.reward_config = {
            'distance_weight': 0.55,
            'speed_weight': 0.2,
            'angle_weight': 0.1,
            'landing_weight': 0.15
        }
        self.reason = "Brak powodu"

    def reset(self):
        """Resetuje środowisko do stanu początkowego."""
        self.lander = Lander.MoonLander(get_random_lander_start_pos(), LANDER_SIZE)
        self.background = Background.StarryBackground(WIDTH, HEIGHT, num_stars=200)  # Regeneruj tło
        self.moon_surface = Surface.MoonSurface(WIDTH, HEIGHT, max_height=30, roughness=30)
        self.lander_velocity = np.array([0, 0], dtype=np.float32)
        self.lander_angle = 0  # Kąt nachylenia statku w stopniach
        self.wind = 0  # Początkowy wiatr
        self.done = False
        self.reward = 0.0
        self.reason = "Brak powodu"
        return self.get_state()

    def step(self, action):
        rotation = 0
        thrust = 0
        left_action = False
        right_action = False
        self.reward = 0.0

        if action == 1:  # Obrót w lewo
            rotation = -ROTATION_SPEED
            left_action = True
        elif action == 2:  # Obrót w prawo
            rotation = ROTATION_SPEED
            right_action = True
        elif action == 3:  # Ciąg
            thrust = THRUST

        # Aktualizuj lądownik
        self.lander.update(thrust, rotation, GRAVITY, random.uniform(-0.05, 0.05), left_action, right_action)

        # Aktualizuj zmienne stanu lądownika
        self.lander_velocity = self.lander.velocity
        self.lander_angle = self.lander.angle

        # Distance-based reward
        distance_to_platform = abs(self.lander.pos[0] - (PLATFORM_POS[0] + PLATFORM_WIDTH // 2))
        distance_reward = max(0, (PLATFORM_WIDTH / 2 - distance_to_platform)) * self.reward_config['distance_weight']

        # Speed penalty
        vertical_speed_penalty = abs(self.lander_velocity[1])
        speed_penalty = vertical_speed_penalty * self.reward_config['speed_weight'] \
            if self.lander.pos[1] < PLATFORM_POS[1] + 100 else 0.2 * vertical_speed_penalty

        # Angle penalty
        angle_penalty = abs(self.lander_angle) / MAX_LANDING_ANGLE * self.reward_config['angle_weight']

        self.reward += distance_reward
        self.reward -= speed_penalty
        self.reward += angle_penalty

        if self.lander.pos[1] < 0:
            self.done = True
            self.reward -= 100 # Kolizja z sufitem
            self.reason = "Kowboj przesadził i poleciał prosto w gwiazdy! Orbitę opuścił szybciej niż pędzący meteoryt."

        if self.lander.pos[0] < 0 or self.lander.pos[0] + self.lander.size[0] > WIDTH:
            self.done = True
            self.reward -= 100  # Kolizja ze ścianą
            self.reason = "Dziki zachód wymaga dzikiej precyzji, kowboju!"

        if self.moon_surface.check_collision(self.lander):
            self.done = True
            self.reward -= 100  # Zderzenie z powierzchnią księżyca
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

    def _check_landing(self):
        """Sprawdza, czy lądowanie było udane."""
        center_x = PLATFORM_POS[0] + PLATFORM_WIDTH / 2
        distance_to_center = abs(self.lander.pos[0] - center_x)
        too_fast = abs(self.lander_velocity[1]) > MAX_LANDING_SPEED
        too_tilted = abs(self.lander_angle) > MAX_LANDING_ANGLE

        # Smooth reward based on landing quality
        if distance_to_center <= PLATFORM_WIDTH / 2:
            if not too_fast and not too_tilted:
                # Perfect landing
                self.reward += 500 - (distance_to_center + 10 * abs(self.lander_velocity[1]))
                self.reason = "Howdy, Houston! Lądownik zaparkowany prosto jak w saloonie."
            elif too_fast or too_tilted:
                # Imperfect landing
                self.reward += 200 - (50 if too_fast else 0) - (50 if too_tilted else 0)
                self.reason = "Nieidealne lądowanie, ale platforma przeżyła!"
            else:
                # Poor landing
                self.reward -= 50
                self.reason = "Za szybko i za krzywo – lądowanie jak u pijącego szeryfa w miasteczku!"
        else:
            # Missed the platform
            self.reward -= 100
            self.reason = "Kowboj zgubił platformę! To nie dziki zachód, ląduj celniej!"

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
        platform_center_x = PLATFORM_POS[0] + PLATFORM_WIDTH / 2
        platform_y = PLATFORM_POS[1]  # Wysokość platformy
        return np.array([
            self.lander.pos[0],  # Pozycja X lądownika
            self.lander.pos[1],  # Pozycja Y lądownika
            self.lander_velocity[0],  # Prędkość X
            self.lander_velocity[1],  # Prędkość Y
            self.lander_angle,  # Kąt lądownika
        ], dtype=np.float32)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)


    with open('rewards.csv', 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        if rows:
            last_episode = int(rows[-1][0])
        else:
            last_episode = 0

    if last_episode == 0:
        with open('rewards.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Reward"])


    env = MoonLanderEnv()

    # Set up the agent
    state_size = 5  # state vector length
    action_size = 4  # number of actions
    clock = pygame.time.Clock()
    agent = DQNAgent.DQNAgent(
        state_size=5,
        action_size=4,
        device=device,
        epsilon_start=1.0,  # Start with full exploration
        epsilon_end=0.01,  # Minimum exploration rate
        epsilon_decay_rate=0.9999,  # Very slow decay for long training
    )

    # Load saved model if exists
    if os.path.exists("lander_pytorch.pth"):
        print("Loading saved model...")
        agent.load("lander_pytorch.pth")

    episodes = 5000
    batch_size = 32

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0

        for time in range(500):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            action = agent.act(state)
            next_state, reward, reason, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                print(
                    f"Episode {e + 1}/{episodes} - Total reward: {total_reward:.2f} - Reason: {reason} - Epsilon: {agent.epsilon:.2f}")
                with open('rewards.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([last_episode+ e + 1, total_reward])
                break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    print("Saving trained model...")
    agent.save('lander_pytorch.pth')

    while True:
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            action = agent.act(state)
            next_state, reward, reason, done, _ = env.step(action)

            env.render()
            state = np.reshape(next_state, [1, state_size])

            clock.tick(30)  # Limit to 30 FPS

            if done:
                print(f"Episode complete! Reward: {reward} - Reason: {reason}")
                break


if __name__ == "__main__":
    main()
