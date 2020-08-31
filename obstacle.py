import random


class Obstacle:
    def __init__(self, centerx, centery, radius):
        self.centerx = centerx
        self.centery = centery
        self.radius = radius

    def contains(self, x, y):
        square_dist = (self.centerx - x) ** 2 + (self.centery - y) ** 2
        return square_dist < self.radius ** 2


class ObstacleGenerator:
    def __init__(self, num_obstacles, arena_side_length):
        self.num_obstacles = num_obstacles
        self.arena_side_length = arena_side_length
        obstacles = []
        for i in range(self.num_obstacles):
            centerx, centery = self.get_random_point()
            radius = random.random() * (self.arena_side_length ** 0.5) / 4
            obstacle = Obstacle(centerx, centery, radius)
            obstacles.append(obstacle)

        self.obstacle_array = obstacles

    def get_random_point(self):
        x = random.randint(2, self.arena_side_length - 2)
        y = random.randint(2, self.arena_side_length - 2)
        return x, y
