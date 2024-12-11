import random
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
