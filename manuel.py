import sys
import pygame
import torch

import MoonLanderEnv

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Set up the agent
    state_size = 5  # state vector length
    episodes = 1000
    clock = pygame.time.Clock()
    env = MoonLanderEnv.MoonLanderEnv()

    outcome = {
        0:"Brak powodu",
        1:"Howdy, Houston! Lądownik zaparkowany prosto jak w saloonie.",
        2:"Zwolnij kowboju! Platforma ledwo ustała!",
        3:"Krzywo jak dach stodoły po burzy, ale się trzyma!",
        4:"Za szybko i za krzywo – lądowanie jak u pijącego szeryfa w miasteczku!",
        5:"Kowboj przesadził i poleciał prosto w gwiazdy! Orbitę opuścił szybciej niż pędzący meteoryt.",
        6:"Dziki zachód wymaga dzikiej precyzji, kowboju!",
        7:"Kowboj wpadł na skały jak kaktus w burzę piaskową!",
        8:"Czas się skończył, kowboju! Nie wykonano misji na czas."
    }

    # Inicjalizacja Pygame
    pygame.init()
    pygame.display.set_mode((600, 400), flags=pygame.SHOWN)

    while True:
        state = env.reset()
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    env.plot.plot_rewards()
                    sys.exit()

            # Wybór akcji
            keys = pygame.key.get_pressed()
            action = 0
            if keys[pygame.K_LEFT]:
                action = 1
            elif keys[pygame.K_RIGHT]:
                action = 2
            elif keys[pygame.K_UP]:
                action = 3

            next_state, reward, reason, done = env.step(action)

            env.render()

            clock.tick(30)  # Limit to 30 FPS

            if done:
                outcome_string = outcome.get(reason, "Nieznany powód")
                print(f"Episode complete! Reward: {reward} - Reason: {outcome_string}")
                break

if __name__ == "__main__":
    main()
