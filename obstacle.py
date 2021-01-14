import random


class Obstacle:
    def __init__(self, centerx, centery, radius, centerz=None):
        self.centerx = centerx
        self.centery = centery
        self.centerz = centerz
        self.radius = radius

    def contains(self, x, y, z=None):
        if z is None:
            square_dist = (self.centerx - x) ** 2 + (self.centery - y) ** 2
        else:
            square_dist = (self.centerx - x) ** 2 + (self.centery - y) ** 2 + (self.centerz - z) ** 2
        return square_dist < self.radius ** 2


class ObstacleGenerator:
    def __init__(self, num_obstacles, arena_side_length, do_3d=False):
        self.num_obstacles = num_obstacles
        self.arena_side_length = arena_side_length
        obstacles = []
        for i in range(self.num_obstacles):
            centerx, centery, centerz = self.get_random_point()
            if not do_3d:
                centerz = None
            radius = random.random() * (self.arena_side_length ** 0.5) / 2
            obstacle = Obstacle(centerx, centery, radius, centerz)
            obstacles.append(obstacle)

        self.obstacle_array = obstacles

    def get_random_point(self):
        x = random.randint(2, self.arena_side_length - 2)
        y = random.randint(2, self.arena_side_length - 2)
        z = random.randint(2, self.arena_side_length - 2)
        return x, y, z

    def get_obstacles(self):
        return self.obstacle_array
