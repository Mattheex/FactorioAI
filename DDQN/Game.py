import numpy as np
import pygame
import sys
import gymnasium as gym
from typing import Optional
from gymnasium.spaces import Tuple, Box

from entities import Belt, Case, Supplier, Vendor, Maker, ItemList
from var import TOP, RIGHT, DOWN, LEFT, WIDTH, HEIGHT, CELL_SIZE, FONT_SIZE, BACKGROUND_COLOR, TEXT_COLOR, \
    CELL_NUMBER, ITEM_SIZE

"""def twoDprint(A):
  
    for i in A:
        print("\t".join(map(str, i)))


def get_neighbors(array, row, col):
  
    rows, cols = array.shape

    row_start = max(0, row - 1)
    row_end = min(rows, row + 2)
    col_start = max(0, col - 1)
    col_end = min(cols, col + 2)

    neighbors = array[row_start:row_end, col_start:col_end]

    center_position = np.array([row - row_start, col - col_start])

    return neighbors, center_position"""

pygame.init()

# Set up the pygame window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Array Display")
font = pygame.font.Font(None, FONT_SIZE)


# Main loop


def decode_id(unique_id):
    a = unique_id // (5 * 4 * 4)
    b = (unique_id % (5 * 4 * 4)) // (4 * 4)
    c = (unique_id % (4 * 4)) // 4
    d = unique_id % 4
    return a, b, c, d


class Game(gym.Env):
    def __init__(self):
        super(Game, self).__init__()
        self.n = CELL_NUMBER
        self.earn = 0
        self.grid = np.empty((self.n, self.n)).tolist()
        """self.items = ItemList()
        self.exemple_game()"""
        self.running = True
        self.clock = pygame.time.Clock()
        self.tolerance = 1.8
        self.action_space = gym.spaces.Discrete(400)
        self.observation_space = Box(low=0, high=4, shape=(self.n, self.n), dtype=np.int32)

    def add_items(self, x, y, obj, direction=None):
        coor = np.array((x, y))
        self.grid[x][y] = obj(self, coor, direction=direction)

    def exemple_game(self):
        for x in range(self.n):
            for y in range(self.n):
                self.add_items(x, y, Case)
        """self.add_items(0, 1, Supplier, DOWN)
        self.add_items(0, 2, Belt, RIGHT)
        self.add_items(1, 2, Belt, RIGHT)
        self.add_items(2, 2, Belt, DOWN)
        self.add_items(2, 3, Vendor)"""

    def case_not_null(self):
        return sum(1 for obj in sum(self.grid, []) if type(obj) is not Case)

    def step(self, action):
        a, b, c, d = decode_id(action)
        pre_earn = self.earn

        if a == 4 and b == 4 and c == 3 and d == 3:
            pass
        else:
            # print(a, b, c, d)
            rot_map = [LEFT, TOP, DOWN, RIGHT]
            cell_map = [Case, Belt, Supplier, Vendor]
            self.add_items(a, b, cell_map[c], rot_map[d])

        # 0 0 0 0 -> Case (0,0)
        # 0 0 0 1 ->
        # 1 0 1 1 -> Belt (1,0) LEFT
        # 4 4 3 0 -> Vendor (4,4)

        # pre_earn = self.earn
        delta = self.clock.tick()
        # delta = 6
        for obj in sum(self.grid, []):
            for item in self.items:
                if not item.action:
                    if isinstance(obj, Belt):
                        dist = abs(obj.distance(item))
                        if dist < self.tolerance:
                            if item not in obj.items:
                                item.set_vector(obj.speed_vector)
                                item.set_center(obj.get_center())
                                obj.items.append(item)
                                item.action = True
                        else:
                            if dist < obj.size:
                                item.action = True
                            if item in obj.items:
                                obj.items.remove(item)

                    elif isinstance(obj, Vendor):
                        dist = abs(obj.distance(item))
                        if dist < self.tolerance:
                            obj.sell(item)
                            self.items.remove(item)
                        if dist < obj.size:
                            item.action = True

                    elif type(obj) is Case:
                        dist = abs(obj.distance(item))
                        if dist < self.tolerance:
                            self.items.remove(item)
                        if dist < obj.size:
                            item.action = True

            obj.turn()

        for item in self.items:
            item.move(delta)
            item.action = False

        # An environment is completed if and only if the agent has reached the target
        terminated = self.earn >= 30
        truncated = False
        reward = (self.earn - pre_earn) * 50 - self.case_not_null()
        observation = self._get_obs()
        info = {}

        return observation, reward, terminated, truncated, info

    def board_to_image(self):
        img = np.zeros((HEIGHT, WIDTH), dtype=int)
        for obj in sum(self.grid, []):
            if type(obj) is not Case:
                coor = obj.coor.astype(int)
                img[coor[0]:coor[0] + obj.size, coor[1]:coor[1] + obj.size] = obj.class_id
        for item in self.items:
            coor = item.coor.astype(int)
            img[coor[0]:coor[0] + item.size, coor[1]:coor[1] + item.size] = item.class_id

        trunc = round(ITEM_SIZE / 2)
        img = img[::trunc, ::trunc]

        """for i in img[0]:
            print(" ".join(map(str, i)))"""
        img = np.expand_dims(img, axis=0)
        return img

    def _get_obs(self):
        return self.board_to_image()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        """# Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()"""
        self.exemple_game()
        self.items = ItemList()
        self.clock = pygame.time.Clock()
        self.earn = 0

        observation = self._get_obs()

        return observation, {}

    def render(self):
        screen.fill(BACKGROUND_COLOR)

        # agent.turn(self.grid,self.items)

        text_surface = font.render(str(self.earn), True, TEXT_COLOR)
        x = 10
        y = 10
        screen.blit(text_surface, (x, y))

        # agent.observe(g)

        for obj in sum(self.grid, []):
            screen.blit(obj.img, tuple(obj.coor))

        for item in self.items:
            screen.blit(item.img, tuple(item.coor))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.flip()


"""g = Game()
agent = Agent(g)
g.run()"""
# states = np.array([[0 for _ in range(n)] for _ in range(n)])

# agent = Agent(states)

# print(agent.reward(g))
