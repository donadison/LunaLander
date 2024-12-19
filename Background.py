import random
import pygame
import math


class StarryBackground:
    def __init__(self, width, height, num_stars, shooting_star_frequency=0.01):
        # Konstruktor inicjalizujący tło pełne gwiazd
        self.width = width  # Szerokość okna gry
        self.height = height  # Wysokość okna gry
        self.num_stars = num_stars  # Liczba gwiazd do wygenerowania
        self.shooting_star_frequency = shooting_star_frequency  # Szansa na pojawienie się spadającej gwiazdy w każdej klatce
        self.stars = self._generate_stars()  # Generacja gwiazd
        self.supernovas = []  # Lista aktywnych supernowych
        self.shooting_stars = []  # Lista spadających gwiazd

    def _generate_stars(self):
        """
        Generuje listę gwiazd z losowymi pozycjami i jasnością.
        """
        return [
            {
                "pos": (random.randint(0, self.width), random.randint(0, self.height)),  # Losowa pozycja gwiazdy
                "brightness": random.randint(100, 255),  # Losowa jasność dla efektu migotania
                "color": random.choice([  # Losowy kolor spośród dwóch opcji
                    (random.randint(200, 255), random.randint(200, 255), random.randint(255, 255)),  # Niebieskawy
                    (random.randint(255, 255), random.randint(200, 255), random.randint(100, 200))  # Żółtawy
                ]),
                "is_supernova": False  # Flaga wskazująca, czy ta gwiazda to supernowa
            }
            for _ in range(self.num_stars)  # Wygeneruj `num_stars` gwiazd
        ]

    def _create_supernova(self, pos):
        """
        Tworzy nową supernową w podanej pozycji.
        """
        return {
            "pos": pos,  # Pozycja supernowej
            "size": 2,  # Początkowy rozmiar supernowej
            "expansion_rate": random.uniform(0.1, 0.3),  # Tempo rozszerzania się supernowej
            "lifetime": 30  # Czas trwania supernowej w klatkach
        }

    def _create_shooting_star(self):
        """
        Tworzy nową spadającą gwiazdę z losową pozycją, kierunkiem i długością.
        """
        start_x = random.randint(0, self.width)  # Losowa pozycja startowa X
        start_y = random.randint(0, self.height // 2)  # Spadająca gwiazda zaczyna w górnej połowie ekranu
        length = random.randint(30, 100)  # Długość spadającej gwiazdy
        angle = random.uniform(math.radians(30), math.radians(120))  # Losowy kąt w zakresie 30° do 120°
        speed = random.uniform(2, 5)  # Prędkość spadającej gwiazdy
        return {
            "start_pos": (start_x, start_y),  # Początkowa pozycja spadającej gwiazdy
            "length": length,  # Długość spadającej gwiazdy
            "angle": angle,  # Kąt kierunku spadającej gwiazdy
            "speed": speed,  # Prędkość spadającej gwiazdy
            "timer": random.randint(30, 60)  # Czas życia spadającej gwiazdy w klatkach
        }

    def update(self):
        """
        Aktualizuje gwiazdy, supernowe i spadające gwiazdy do animacji.
        """
        # Aktualizuj gwiazdy dla efektu migotania
        for star in self.stars:
            # Nieznacznie zmień jasność dla migotania
            star["brightness"] += random.choice([-10, 10])  # Losowo zmienia jasność
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
            shooting_star["timer"] -= 1  # Zmniejsz czas życia spadającej gwiazdy
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
                pygame.draw.circle(screen,  # Narysuj okręgi dla gwiazd
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
            pygame.draw.line(screen, (255, 255, 255), start_pos, end_pos, 2)  # Narysuj linię spadającej gwiazdy
