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

import random
import math
import pygame


class MoonSurface:
    def __init__(self, width, height, max_height, roughness):
        self.width = width
        self.height = height
        self.max_height = max_height
        self.roughness = roughness
        self.surface_points = self._generate_surface()

    def _generate_surface(self):
        """Generuje powierzchnię księżyca z losowymi wysokościami."""
        points = [(0, self.height)]
        x_step = self.width // 20  # Dzielimy powierzchnię na segmenty

        for x in range(0, self.width + x_step, x_step):
            y = self.height - random.randint(0, self.max_height)
            points.append((x, y))

        points.append((self.width, self.height))  # Zamykamy powierzchnię
        return points

    def render(self, screen):
        """Renderuje powierzchnię księżyca."""
        pygame.draw.polygon(screen, (255, 255, 194), self.surface_points)

    def check_collision(self, lander):
        """Sprawdza, czy lądownik zderzył się z powierzchnią."""
        for i in range(len(self.surface_points) - 1):
            p1 = self.surface_points[i]
            p2 = self.surface_points[i + 1]

            # Sprawdzamy, czy lądownik przecina segment powierzchni
            if self._line_intersects_lander(p1, p2, lander):
                return True
        return False

    def _line_intersects_lander(self, p1, p2, lander):
        """Sprawdza, czy segment linii przecina prostokąt lądownika."""
        lander_box = [
            (lander.pos[0] - lander.size[0] / 2, lander.pos[1] - lander.size[1] / 2),
            (lander.pos[0] + lander.size[0] / 2, lander.pos[1] - lander.size[1] / 2),
            (lander.pos[0] + lander.size[0] / 2, lander.pos[1] + lander.size[1] / 2),
            (lander.pos[0] - lander.size[0] / 2, lander.pos[1] + lander.size[1] / 2),
        ]

        for i in range(len(lander_box)):
            lp1 = lander_box[i]
            lp2 = lander_box[(i + 1) % len(lander_box)]
            if self._line_intersects(p1, p2, lp1, lp2):
                return True
        return False

    def _line_intersects(self, p1, p2, q1, q2):
        """Sprawdza, czy dwa segmenty linii się przecinają."""

        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)


class StarryBackground:
    def __init__(self, width, height, num_stars, shooting_star_frequency=0.01):
        self.width = width
        self.height = height
        self.num_stars = num_stars
        self.shooting_star_frequency = shooting_star_frequency  # Szansa na pojawienie się spadającej gwiazdy w każdej klatce
        self.stars = self._generate_stars()
        self.supernovas = []  # Lista do przechowywania aktywnych supernowych
        self.shooting_stars = []  # Lista do przechowywania spadających gwiazd

    def _generate_stars(self):
        """
        Generuje listę gwiazd z losowymi pozycjami i jasnością.
        """
        return [
            {
                "pos": (random.randint(0, self.width), random.randint(0, self.height)),
                "brightness": random.randint(100, 255),  # Losowa jasność dla efektu migotania
                "color": random.choice([
                    (random.randint(200, 255), random.randint(200, 255), random.randint(255, 255)),  # Niebieskawy
                    (random.randint(255, 255), random.randint(200, 255), random.randint(100, 200))  # Żółtawy
                ]),
                "is_supernova": False  # Flaga wskazująca, czy ta gwiazda to supernowa
            }
            for _ in range(self.num_stars)
        ]

    def _create_supernova(self, pos):
        """
        Tworzy nową supernową w podanej pozycji.
        """
        return {
            "pos": pos,
            "size": 2,  # Początkowy rozmiar supernowej
            "expansion_rate": random.uniform(0.1, 0.3),  # Tempo rozszerzania się
            "lifetime": 30  # Czas trwania supernowej w klatkach
        }

    def _create_shooting_star(self):
        """
        Tworzy nową spadającą gwiazdę z losową pozycją, kierunkiem i długością.
        """
        start_x = random.randint(0, self.width)
        start_y = random.randint(0, self.height // 2)  # Spadająca gwiazda zaczyna w górnej połowie ekranu
        length = random.randint(30, 100)
        angle = random.uniform(math.radians(30), math.radians(120))
        speed = random.uniform(2, 5)  # Prędkość spadającej gwiazdy
        return {
            "start_pos": (start_x, start_y),
            "length": length,
            "angle": angle,
            "speed": speed,
            "timer": random.randint(30, 60)  # Spadająca gwiazda trwa od 30 do 60 klatek
        }

    def update(self):
        """
        Aktualizuje gwiazdy, supernowe i spadające gwiazdy do animacji.
        """
        # Aktualizuj gwiazdy dla efektu migotania
        for star in self.stars:
            # Nieznacznie zmień jasność dla migotania
            star["brightness"] += random.choice([-10, 10])
            star["brightness"] = max(100, min(255, star["brightness"]))  # Ogranicz jasność

            # Losowo zamień gwiazdę w supernową
            if not star["is_supernova"] and random.random() < 0.0003:  # Mała szansa na supernową
                star["is_supernova"] = True
                self.supernovas.append(self._create_supernova(star["pos"]))

        # Aktualizuj supernowe
        for supernova in self.supernovas:
            supernova["size"] += supernova["expansion_rate"]  # Rozszerz supernową
            supernova["lifetime"] -= 1  # Zmniejsz czas trwania

        # Usuń wygasłe supernowe
        self.supernovas = [s for s in self.supernovas if s["lifetime"] > 0]

        # Dodaj nowe spadające gwiazdy z małym prawdopodobieństwem w każdej klatce
        if random.random() < self.shooting_star_frequency:
            self.shooting_stars.append(self._create_shooting_star())

        # Aktualizuj spadające gwiazdy (przesuń je)
        for shooting_star in self.shooting_stars:
            shooting_star["timer"] -= 1
            shooting_star["start_pos"] = (
                shooting_star["start_pos"][0] + shooting_star["speed"] * math.cos(shooting_star["angle"]),
                shooting_star["start_pos"][1] + shooting_star["speed"] * math.sin(shooting_star["angle"])
            )

        # Usuń wygasłe spadające gwiazdy
        self.shooting_stars = [s for s in self.shooting_stars if s["timer"] > 0]

    def render(self, screen):
        """
        Renderuje gwiazdy, supernowe i spadające gwiazdy na ekranie.
        """
        # Renderuj normalne gwiazdy
        for star in self.stars:
            if not star["is_supernova"]:
                pygame.draw.circle(screen,
                                   (star["color"][0] * (star["brightness"] / 255),
                                    # Dostosuj kolor na podstawie jasności
                                    star["color"][1] * (star["brightness"] / 255),
                                    star["color"][2] * (star["brightness"] / 255)), star["pos"], 2)

        # Renderuj supernowe jako rozszerzające się ośmiokąty
        for supernova in self.supernovas:
            pos = supernova["pos"]
            size = supernova["size"]

            # Oblicz punkty dla ośmiokąta
            points = []
            for i, angle in enumerate(range(0, 360, 45)):  # Ośmiokąt z 8 punktami
                expansion_factor = 2 if i % 2 == 0 else 0.5  # Rozszerzaj tylko co drugi punkt
                expanded_size = size * expansion_factor
                points.append((pos[0] + expanded_size * math.cos(math.radians(angle)),
                               pos[1] + expanded_size * math.sin(math.radians(angle))))

            # Narysuj ośmiokąt
            pygame.draw.polygon(screen, (255, 255, 0), points)

        # Renderuj spadające gwiazdy jako linie
        for shooting_star in self.shooting_stars:
            start_pos = shooting_star["start_pos"]
            end_pos = (
                start_pos[0] + shooting_star["length"] * math.cos(shooting_star["angle"]),
                start_pos[1] + shooting_star["length"] * math.sin(shooting_star["angle"])
            )
            pygame.draw.line(screen, (255, 255, 255), start_pos, end_pos, 2)


class MoonLander:
    def __init__(self, start_pos, size):
        self.pos = np.array(start_pos, dtype=np.float32)
        self.velocity = np.array([0, 0], dtype=np.float32)
        self.angle = 0  # Początkowy kąt w stopniach
        self.size = size
        self.points = [
            (0, -size[1] / 2),  # Górny punkt (0)
            (-size[0] / 2, size[1] / 4),  # Lewy środek lądownika (1)
            (-size[0] / 3, size[1] / 2),  # Dolny lewy punkt (2)
            (size[0] / 3, size[1] / 2),  # Dolny prawy punkt (3)
            (size[0] / 2, size[1] / 4)  # Prawy środek lądownika (4)
        ]
        self.left_thrust_active = False
        self.right_thrust_active = False

        # Parametry silnika
        self.thrust_strength = 0.0  # Aktualna siła ciągu (0 - 1)
        self.left_thrust_strength = 0.0  # Aktualna siła ciągu lewego (0 - 1)
        self.right_thrust_strength = 0.0  # Aktualna siła ciągu prawego (0 - 1)
        self.max_thrust_strength = 1.0  # Maksymalna siła ciągu

    def update(self, thrust, rotation, gravity, wind, left_thrust=False, right_thrust=False):
        """
        Aktualizuje pozycję, prędkość i kąt lądownika na podstawie sił wejściowych.
        """
        # Zastosuj rotację
        self.angle += rotation

        # Ciąg powinien być zastosowany w kierunku kąta lądownika
        if thrust != 0:
            # Oblicz składowe x i y ciągu na podstawie kąta lądownika
            rad_angle = math.radians(self.angle)
            thrust_x = math.sin(-rad_angle) * thrust  # Składowa pozioma
            thrust_y = math.cos(-rad_angle) * thrust  # Składowa pionowa
            # Zaktualizuj prędkość na podstawie składowych ciągu
            self.velocity[0] += thrust_x  # Zastosuj ciąg poziomy
            self.velocity[1] += thrust_y  # Zastosuj ciąg pionowy

            # Stopniowo zwiększaj siłę ciągu do maksymalnej wartości
            self.thrust_strength = min(self.max_thrust_strength, self.thrust_strength + 0.1)
        else:
            # Stopniowo zmniejszaj siłę ciągu, gdy ciąg wynosi 0
            self.thrust_strength = max(0, self.thrust_strength - 0.1)

        if left_thrust:
            self.left_thrust_active = True
            self.left_thrust_strength = min(self.max_thrust_strength, self.left_thrust_strength + 0.1)
        else:
            self.left_thrust_active = False
            self.left_thrust_strength = max(0, self.left_thrust_strength - 0.1)

        if right_thrust:
            self.right_thrust_active = True
            self.right_thrust_strength = min(self.max_thrust_strength, self.right_thrust_strength + 0.1)
        else:
            self.right_thrust_active = False
            self.right_thrust_strength = max(0, self.right_thrust_strength - 0.1)

        # Zastosuj grawitację i wiatr
        self.velocity[1] += gravity
        self.velocity[0] += wind

        # Zaktualizuj pozycję
        self.pos += self.velocity

    def render(self, screen):
        """
        Renderuje lądownik na ekranie.
        """
        rad_angle = math.radians(self.angle)
        cos_theta = math.cos(rad_angle)
        sin_theta = math.sin(rad_angle)

        # Obróć i przesuń punkty
        rotated_points = [
            (
                int(self.pos[0] + x * cos_theta - y * sin_theta),
                int(self.pos[1] + x * sin_theta + y * cos_theta)
            )
            for x, y in self.points
        ]

        # Upewnij się, że rotated_points są poprawnymi krotkami liczb całkowitych
        if len(rotated_points) > 0 and all(isinstance(pt, tuple) and len(pt) == 2 for pt in rotated_points):
            self._render_thruster(screen, rotated_points)
            self._render_side_thrusters(screen, rotated_points)
            pygame.draw.polygon(screen, ORANGE, rotated_points)
            pygame.draw.lines(screen, GRAY, True, rotated_points, 3)  # Obrys
            pygame.draw.line(screen, GRAY, rotated_points[1], rotated_points[3], 3)  # Dekoracja
            pygame.draw.line(screen, GRAY, rotated_points[1], rotated_points[4], 3)  # Dekoracja
            pygame.draw.line(screen, GRAY, rotated_points[2], rotated_points[4], 3)  # Dekoracja
        else:
            print("Błąd: Wykryto nieprawidłowe punkty w rotated_points.")

    def _render_thruster(self, background, rotated_points):
        """
        Renderuje efekt działania silnika na podstawie siły ciągu, umiejscowiony na dolnym środku lądownika.
        """
        if self.thrust_strength > 0:
            # Oblicz dolny środek lądownika na podstawie obróconych punktów
            bottom_left = rotated_points[1]
            bottom_right = rotated_points[4]
            thruster_pos = (
                (bottom_left[0] + bottom_right[0]) / 2,
                (bottom_left[1] + bottom_right[1]) / 2
            )

            # Dostosuj rozmiar i kolor na podstawie siły ciągu
            max_size = 25  # Maksymalny rozmiar płomienia
            thruster_size = max_size * self.thrust_strength
            color = (0, 191, int(255 * self.thrust_strength))  # Odcienie niebieskiego w zależności od siły

            # Oblicz kąt płomienia silnika na podstawie rotacji lądownika
            rad_angle = math.radians(self.angle)
            flame_offset = thruster_size * 1.5  # Przesunięcie płomienia w dół

            # Oblicz pozycję płomienia na podstawie rotacji
            flame_pos = (
                thruster_pos[0] + flame_offset * math.sin(-rad_angle),
                thruster_pos[1] + flame_offset * math.cos(-rad_angle)
            )

            # Narysuj płomień silnika
            pygame.draw.line(
                background,
                color,
                thruster_pos,
                flame_pos,
                width=5  # Szerokość płomienia
            )

            # Dodaj efekt poświaty z małym okręgiem
            pygame.draw.circle(
                background,
                (0, 191, int(255 * self.thrust_strength)),
                (int(flame_pos[0]), int(flame_pos[1])),
                int(thruster_size * 0.2)
            )

    def _render_side_thrusters(self, background, rotated_points):
        """
        Renderuje boczne silniki, jeśli są aktywne, z płomieniami obróconymi o 45 stopni.
        """
        if self.left_thrust_active or self.right_thrust_active:
            # Oblicz pozycje bocznych silników
            left_thruster_pos = rotated_points[1]
            right_thruster_pos = rotated_points[4]

            # Parametry płomieni silników
            max_flame_length = 15  # Maksymalna długość płomienia
            left_flame_strength = self.left_thrust_strength
            right_flame_strength = self.right_thrust_strength

            # Kąt rotacji landera w radianach
            rad_angle = math.radians(self.angle)

            # Korekta kąta dla płomieni (45 stopni od silnika)
            angle_offset = math.radians(45)

            if self.left_thrust_active:
                left_flame_length = max_flame_length * (left_flame_strength - right_flame_strength)
                flame_pos = (
                    left_thruster_pos[0] - left_flame_length * math.cos(-rad_angle + angle_offset),
                    left_thruster_pos[1] + left_flame_length * math.sin(-rad_angle + angle_offset)
                )
                flame_color = (0, 191, int(255 * left_flame_strength))  # Dopasowanie koloru centralnego silnika
                pygame.draw.line(background, flame_color, left_thruster_pos, flame_pos, 4)

            if self.right_thrust_active:
                right_flame_length = max_flame_length * (right_flame_strength - left_flame_strength)
                flame_pos = (
                    right_thruster_pos[0] + right_flame_length * math.cos(rad_angle + angle_offset),
                    right_thruster_pos[1] + right_flame_length * math.sin(rad_angle + angle_offset)
                )
                flame_color = (0, 191, int(255 * right_flame_strength))  # Dopasowanie koloru centralnego silnika
                pygame.draw.line(background, flame_color, right_thruster_pos, flame_pos, 4)


# Klasa środowiska
class MoonLanderEnv:
    def __init__(self):
        self.lander = MoonLander(LANDER_START_POS, LANDER_SIZE)
        self.background = StarryBackground(WIDTH, HEIGHT, num_stars=200)  # 200 gwiazd
        self.moon_surface = MoonSurface(WIDTH, HEIGHT, max_height=30, roughness=30)
        self.lander_velocity = np.array([0, 0], dtype=np.float32)
        self.lander_angle = 0
        self.wind = 0
        self.done = False
        self.reward = 0
        self.reason = "Brak powodu"

    def reset(self):
        """Resetuje środowisko do stanu początkowego."""
        self.lander = MoonLander(LANDER_START_POS, LANDER_SIZE)
        self.background = StarryBackground(WIDTH, HEIGHT, num_stars=200)  # Regeneruj tło
        self.moon_surface = MoonSurface(WIDTH, HEIGHT, max_height=30, roughness=30)
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
