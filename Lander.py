import numpy as np
import math
import pygame

GRAY = (128, 128, 128)
ORANGE = (255, 69, 0)


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

    def update(self, thrust, rotation, gravity, left_thrust=False, right_thrust=False):
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

        if len(rotated_points) > 0 and all(isinstance(pt, tuple) and len(pt) == 2 for pt in rotated_points):
            self._render_thruster(screen, rotated_points)
            self._render_side_thrusters(screen, rotated_points)
            pygame.draw.polygon(screen, ORANGE, rotated_points)
            pygame.draw.lines(screen, GRAY, True, rotated_points, 3)  # Obrys
            pygame.draw.line(screen, GRAY, rotated_points[1], rotated_points[3], 2)  # Dekoracja
            pygame.draw.line(screen, GRAY, rotated_points[1], rotated_points[4], 2)  # Dekoracja
            pygame.draw.line(screen, GRAY, rotated_points[2], rotated_points[4], 2)  # Dekoracja
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
            max_flame_length = 20  # Maksymalna długość płomienia
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
