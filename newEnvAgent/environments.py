import numpy as np


class CircularObstacle2D:
    """Static circular obstacle."""
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

    def check_collision(self, agent_x, agent_y, agent_radius, obstacle_buffer=0.0):
        dx = agent_x - self.x
        dy = agent_y - self.y
        combined = agent_radius + self.r + obstacle_buffer
        return dx * dx + dy * dy < combined * combined


class RectangleObstacle2D:
    """Static axis-aligned rectangular obstacle. (x, y) is the center."""
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def check_collision(self, agent_x, agent_y, agent_radius, obstacle_buffer=0.0):
        half_w = self.w / 2
        half_h = self.h / 2
        closest_x = max(self.x - half_w, min(agent_x, self.x + half_w))
        closest_y = max(self.y - half_h, min(agent_y, self.y + half_h))
        dx = agent_x - closest_x
        dy = agent_y - closest_y
        r = agent_radius + obstacle_buffer
        return dx * dx + dy * dy < r * r


class RectangularEnvironment:
    """
    Continuous 2D rectangular environment with circular and rectangular obstacles.

    Coordinate convention: origin at bottom-left corner.
      x ∈ [0, length],  y ∈ [0, breadth]

    This mirrors MAPF's SquareEnvironment but is named more explicitly and supports
    only 2D obstacles (no dynamic agents) for now.

    Attributes
    ----------
    env_start : np.ndarray (2,)
        Bottom-left corner — always [0, 0].
    size : np.ndarray (2,)
        [length, breadth] of the environment.
    obstacles : list
        List of CircularObstacle2D / RectangleObstacle2D objects.
    obstacle_buffer : float
        Small padding added around all obstacles during collision checks.
    boundary_buffer : float
        Minimum clearance from the environment boundary.
    static_circular_obstacles : np.ndarray (N, 3)
        Vectorised circular obstacles: each row is (cx, cy, r).
    static_rectangular_obstacles : np.ndarray (M, 4)
        Vectorised rectangular obstacles: each row is (x_min, x_max, y_min, y_max).
    """

    def __init__(self, length, breadth, obstacles, obs_buffers=True):
        self.env_start = np.array([0.0, 0.0], dtype=np.float64)
        self.size = np.array([length, breadth], dtype=np.float64)
        self.boundary_buffer = 0.1
        self.obstacles = list(obstacles)
        self.obstacle_buffer = 0.0125 * (length + breadth) / 2.0
        if not obs_buffers:
            self.obstacle_buffer = 0.0
        self.static_circular_obstacles = None
        self.static_rectangular_obstacles = None
        self.env_dim = 2
        self._rebuild_vectorized_obstacles()

    def _rebuild_vectorized_obstacles(self):
        """Recompute the numpy arrays used by numba collision functions."""
        circ = [(o.x, o.y, o.r) for o in self.obstacles if isinstance(o, CircularObstacle2D)]
        self.static_circular_obstacles = (np.array(circ, dtype=np.float64)
                                          if circ else np.zeros((0, 3), dtype=np.float64))

        rect = []
        for o in self.obstacles:
            if isinstance(o, RectangleObstacle2D):
                rect.append([o.x - o.w / 2, o.x + o.w / 2, o.y - o.h / 2, o.y + o.h / 2])
        self.static_rectangular_obstacles = (np.array(rect, dtype=np.float64)
                                             if rect else np.zeros((0, 4), dtype=np.float64))

    def add_obstacle(self, obs):
        self.obstacles.append(obs)
        self._rebuild_vectorized_obstacles()
