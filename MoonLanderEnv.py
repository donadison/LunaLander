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
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

DARK_GRAY = (64, 64, 64)
BLACK = (0, 0, 0)

# Parametry lądownika
LANDER_SIZE = (20, 30)
GRAVITY = 0.1
THRUST = -0.2
ROTATION_SPEED = 3  # Stopnie na klatkę
MAX_LANDING_SPEED = 1
MAX_LANDING_ANGLE = 6  # Maksymalny dopuszczalny kąt nachylenia w stopniach

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
        self.done = False
        self.reward = 0.0
        self.distance = self.calculate_distance()
        # Reward configuration (can be tuned)
        self.reward_config = {
            'distance_weight': 1.0,  # Slight penalty for being far
            'speed_weight': 2.0,  # Increase penalty for high velocity
            'angle_weight': 2.0,  # Increase penalty for poor alignment
            'proximity_bonus': 10.0,  # Larger bonus for being close to the platform
            'alignment_bonus': 25.0,  # Encourage upright landings
            'velocity_bonus': 25.0,  # Encourage slow descents
            'failure_penalty': -100.0,  # Harsher crash penalty
            'too_fast': 50.0,  # Reward for suboptimal landing
            'too_tilted': 50.0,  # Reward for suboptimal landing
            'too_tilted_and_too_fast': 25.0,  # Reduced reward for poor landing
            'perfect_score': 200.0  # Increased perfect landing reward
        }
        self.reason = 0
        self.time_limit = 500  # Maximum steps per episode
        self.step_count = 0  # Step counter
        self.last_action = 0

    def reset(self):
        self.lander = Lander.MoonLander(get_random_lander_start_pos(), LANDER_SIZE)
        self.background = Background.StarryBackground(WIDTH, HEIGHT, num_stars=200)
        self.moon_surface = Surface.MoonSurface(WIDTH, HEIGHT, max_height=30, roughness=30)
        self.done = False
        self.reward = 0.0
        self.step_count = 0
        self.distance = self.calculate_distance()
        self.last_action = 0
        self.reason = 0
        return self.get_state()

    def step(self, action):
        self.last_action = action
        self.step_count += 1

        rotation = 0
        thrust = 0

        if action == 1:  # Rotate left
            rotation = -ROTATION_SPEED
        elif action == 2:  # Rotate right
            rotation = ROTATION_SPEED
        elif action == 3:  # Thrust
            thrust = THRUST

        # Update lander state
        self.lander.update(thrust, rotation, GRAVITY, action == 1, action == 2)

        # Calculate current metrics
        self.distance = self.calculate_distance()

        # Calculate rewards
        self.reward = (
                self.calculate_proximity_reward() +
                self.calculate_velocity_reward() +
                self.calculate_alignment_reward())

        # Check for success or failure
        if self.lander.pos[1] < 0:  # ceiling
            self.done = True
            self.reason = 5
            self.reward += self.reward_config['failure_penalty']

        if self.lander.pos[0] < 0 or self.lander.pos[0] + self.lander.size[0] > WIDTH:  #sides
            self.done = True
            self.reason = 6
            self.reward += self.reward_config['failure_penalty']

        if self.moon_surface.check_collision(self.lander):  # Surface crash
            self.done = True
            self.reason = 7

            # Harsher penalty near the platform
            if self.distance < 50:
                self.reward += self.reward_config['failure_penalty'] * 2
            else:
                self.reward += self.reward_config['failure_penalty']

        # Check side collisions with the platform
        if (
                PLATFORM_POS[1] <= self.lander.pos[1] + self.lander.size[1] <= PLATFORM_POS[1] + PLATFORM_HEIGHT and
                (PLATFORM_POS[0] - self.lander.size[0] < self.lander.pos[0] < PLATFORM_POS[
                    0] or  # Left side of platform
                 PLATFORM_POS[0] + PLATFORM_WIDTH < self.lander.pos[0] + self.lander.size[0] < PLATFORM_POS[
                     0] + PLATFORM_WIDTH + self.lander.size[0]  # Right side of platform
                )
        ):
            self.done = True
            self.reason = 7  # platform side collisions
            self.reward += self.reward_config['failure_penalty']

        if (
                self.lander.pos[1] + self.lander.size[1] >= PLATFORM_POS[1]
                and PLATFORM_POS[0] <= self.lander.pos[0] <= PLATFORM_POS[0] + PLATFORM_WIDTH
        ):
            self.lander.pos[1] = PLATFORM_POS[1] - self.lander.size[1]
            self.done = True
            self._check_landing()

        if self.step_count >= self.time_limit:
            self.done = True
            self.reason = 8
            self.reward += self.reward_config['failure_penalty']

        return self.get_state(), self.reward, self.reason, self.done

    def _check_landing(self):
        too_fast = abs(self.lander.velocity[1]) > MAX_LANDING_SPEED
        too_tilted = abs(self.lander.angle) > MAX_LANDING_ANGLE

        landing_reward = self.reward_config['perfect_score']

        if not too_fast and not too_tilted:  # Perfect landing
            self.reward += landing_reward
            self.reason = 1
        elif too_fast and not too_tilted:  # Too fast but aligned
            self.reward += landing_reward * 0.5
            self.reason = 2
        elif too_tilted and not too_fast:  # Too tilted but slow
            self.reward += landing_reward * 0.5
            self.reason = 3
        else:  # Both too fast and too tilted
            self.reward += landing_reward * 0.25
            self.reason = 4

    def calculate_distance(self):
        center_x = PLATFORM_POS[0] + PLATFORM_WIDTH / 2
        top_y = PLATFORM_POS[1]
        horizontal_distance = abs(self.lander.pos[0] - center_x)
        vertical_distance = abs(self.lander.pos[1] - top_y)
        return np.sqrt(horizontal_distance ** 2 + vertical_distance ** 2)

    def calculate_proximity_reward(self):
        if self.distance < 50:
            return self.reward_config['proximity_bonus'] / (self.distance + 1)  # Higher reward when closer
        return -self.distance * self.reward_config['distance_weight']

    def calculate_velocity_reward(self):
        # Only reward when close to the platform
        if self.distance < 50:
            vel_x, vel_y = self.lander.velocity
            total_velocity = np.sqrt(vel_x ** 2 + vel_y ** 2)

            # Reward slow descent
            if abs(vel_y) < 0.5:
                return self.reward_config['velocity_bonus']
            else:
                # Mild penalty for faster descent but not overly harsh
                return -min(total_velocity, 2) * self.reward_config['speed_weight']
        return 0  # No velocity-related rewards/penalties during flight

    def calculate_alignment_reward(self):
        # Only reward alignment near the platform
        if self.distance < 50:
            alignment_penalty = -abs(self.lander.angle) * self.reward_config['angle_weight']

            # Positive reward for maintaining upright alignment
            if abs(self.lander.angle) < 2:
                alignment_penalty += self.reward_config['alignment_bonus']
            return alignment_penalty
        return 0  # No alignment-related rewards/penalties during flight

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
        font = pygame.font.SysFont(None, 16)
        velocity_text = font.render(
            f"Prędkość X: {self.lander.velocity[0]:.2f}  Prędkość Y: {self.lander.velocity[1]:.2f}", True, WHITE)
        angle_text = font.render(f"Kąt: {self.lander.angle:.2f}", True, WHITE)
        distance_text = font.render(f"Odległość: {self.distance:.2f}", True, WHITE)
        reward_text = font.render(f"Nagroda: {self.reward:.2f}", True, WHITE)
        screen.blit(velocity_text, (10, 10))
        screen.blit(angle_text, (10, 30))
        screen.blit(distance_text, (10, 50))
        screen.blit(reward_text, (10, 70))

        # Draw action symbols
        self.draw_action_symbols()

        pygame.display.flip()

    def preprocess_state(self, state):
        state[0] = state[0] / 300 - 1  # Normalize X position (0 to 600 -> -1 to 1)
        state[1] = state[1] / 200 - 1  # Normalize Y position (0 to 400 -> -1 to 1)
        state[2] = state[2] / 20  # Normalize X velocity (-20 to 20 -> -1 to 1)
        state[3] = state[3] / 15  # Normalize Y velocity (-15 to 15 -> -1 to 1)
        state[4] = state[4] / 360  # Normalize angle (-360 to 360 -> -1 to 1)
        return state

    def get_state(self):
        # Raw state from the environment
        raw_state = np.array([
            self.lander.pos[0],  # X position of the lander
            self.lander.pos[1],  # Y position of the lander
            self.lander.velocity[0],  # X velocity
            self.lander.velocity[1],  # Y velocity
            self.lander.angle,  # Angle of the lander
        ], dtype=np.float32)

        # Preprocess and return the normalized state
        return self.preprocess_state(raw_state)

    def draw_action_symbols(self):
        base_x = 10  # Horizontal offset
        base_y = 90  # Vertical offset

        if self.last_action == 1:  # Rotate Left
            # Draw left arrow
            pygame.draw.polygon(screen, YELLOW, [
                (base_x, base_y),  # Left point
                (base_x + 20, base_y - 10),  # Top point
                (base_x + 20, base_y + 10)  # Bottom point
            ])
        elif self.last_action == 2:  # Rotate Right
            # Draw right arrow
            pygame.draw.polygon(screen, YELLOW, [
                (base_x + 40, base_y),  # Right point
                (base_x + 20, base_y - 10),  # Top point
                (base_x + 20, base_y + 10)  # Bottom point
            ])
        elif self.last_action == 3:  # Thrust
            # Draw flame symbol
            pygame.draw.polygon(screen, RED, [
                (base_x + 70, base_y - 10),  # Tip of the flame
                (base_x + 60, base_y),  # Bottom left
                (base_x + 80, base_y)  # Bottom right
            ])
