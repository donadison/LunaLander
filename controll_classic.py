import pygame
import numpy as np
import sys
import MoonLanderEnv

# Funkcja do ręcznego sterowania robotem
def manual_control():
    # Zdefiniowanie mapy klawiszy
    key_map = {
        pygame.K_LEFT: 1,  # rotate left
        pygame.K_RIGHT: 2,  # rotate right
        pygame.K_DOWN: 3   # thrust
    }
    action = 0  # Default action is no action
    keys = pygame.key.get_pressed()  # Get current keys pressed

    # Check for keys pressed and assign corresponding actions
    if keys[pygame.K_LEFT]:
        action = 1  # Rotate left
    elif keys[pygame.K_RIGHT]:
        action = 2  # Rotate right
    elif keys[pygame.K_DOWN]:
        action = 3  # Thrust

    return action

def main():
    state_size = 5  # state vector length
    clock = pygame.time.Clock()
    env = MoonLanderEnv.MoonLanderEnv()
    
    # Inicjalizacja Pygame
    pygame.init()
    pygame.display.set_mode((600, 400), flags=pygame.SHOWN)
    
    while True:
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    env.plot.plot_rewards()  # Zapisuje wykresy, jeśli chcesz
                    sys.exit()

            # Manual control: Get action based on key press
            action = manual_control()

            # Execute the action in the environment
            next_state, reward, reason, done = env.step(action)

            # Render the environment
            env.render()

            # Update the state
            state = np.reshape(next_state, [1, state_size])

            clock.tick(30)  # Limit to 30 FPS

            if done:
                print(f"Episode complete! Reward: {reward} - Reason: {reason}")
                break

if __name__ == "__main__":
    main()
