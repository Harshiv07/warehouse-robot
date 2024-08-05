import pygame
import math
from queue import PriorityQueue, deque
import time

# Define the mapping of product codes to aisles
product_location_mapping = {
    "01-123456": (3, 4),
    "02-123456": (10, 4),
    "03-123456": (15, 23),
    "04-123456": (15, 6),
    "05-123456": (15, 22),
    "06-123456": (30, 4),
    "07-123456": (36, 7),
    "08-123456": (14, 17),
    "09-123456": (24, 5),
    "10-123456": (24, 24),
    "11-123456": (29, 8),
    "12-123456": (31, 13),
}

# List of product codes to be located
product_codes = [
    "01-123456",
    "02-123456",
    "03-123456",
    "04-123456",
    "05-123456",
    "06-123456",
    "07-123456",
    "08-123456",
    "09-123456",
    "10-123456",
    "11-123456",
    "12-123456",
]

# Pygame Configuration for Robot Navigation
WIDTH = 600
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("Warehouse Robots")

RED = (255, 0, 0)
GREEN = (9, 206, 55)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (128, 128, 128)
ROBOT = (205, 92, 202)
ORANGE = (255, 165, 0)
LBLUE = (102, 178, 255)
LGREY = (96, 96, 96)
TARGET = (255, 255, 0)  # Yellow for target locations


class Spot:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == RED

    def is_open(self):
        return self.color == GREEN

    def is_barrier(self):
        return (
            self.color == BLACK
            or self.color == ORANGE
            or self.color == LBLUE
            or self.color == LGREY
            or self.color == ROBOT
        )

    def is_end(self):
        return self.color == GREEN

    def is_target(self):
        return self.color == TARGET

    def reset(self):
        self.color = WHITE

    def make_start(self):
        self.color = ROBOT

    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    def make_barrier(self):
        self.color = BLACK

    def make_rep(self):
        self.color = ORANGE

    def make_del(self):
        self.color = LBLUE

    def make_cs(self):
        self.color = LGREY

    def make_end(self):
        self.color = GREEN

    def make_target(self):
        self.color = TARGET

    def make_path(self):
        self.color = ROBOT

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        self.neighbors = []
        # DOWN
        if (
            self.row < self.total_rows - 1
            and not grid[self.row + 1][self.col].is_barrier()
        ):
            self.neighbors.append(grid[self.row + 1][self.col])

        # UP
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row - 1][self.col])

        # RIGHT
        if (
            self.col < self.total_rows - 1
            and not grid[self.row][self.col + 1].is_barrier()
        ):
            self.neighbors.append(grid[self.row][self.col + 1])

        # LEFT
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col - 1])

        # upper left
        if (self.col > 0 and self.row > 0) and not grid[self.row - 1][
            self.col - 1
        ].is_barrier():
            self.neighbors.append(grid[self.row - 1][self.col - 1])

        # upper right
        if (self.row < self.total_rows - 1 and self.col > 0) and not grid[self.row + 1][
            self.col - 1
        ].is_barrier():
            self.neighbors.append(grid[self.row + 1][self.col - 1])

        # lower left
        if (self.col < self.total_rows - 1 and self.row > 0) and not grid[self.row - 1][
            self.col + 1
        ].is_barrier():
            self.neighbors.append(grid[self.row - 1][self.col + 1])

        # lower right
        if (
            self.col < self.total_rows - 1 and self.row < self.total_rows - 1
        ) and not grid[self.row + 1][self.col + 1].is_barrier():
            self.neighbors.append(grid[self.row + 1][self.col + 1])

    def __lt__(self, other):
        return False


def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def reconstruct_path(came_from, current, draw, final, begin):
    res = []
    while current in came_from:
        current = came_from[current]
        res.append(current)
    res1 = res[::-1]
    res1.append(final)

    for curr in res1:
        if curr == begin:
            curr.color = WHITE
        f = False

        if curr.color == GREEN and curr != final:
            f = True

        if curr.is_barrier():
            continue

        curr.make_path()
        time.sleep(0.3)
        draw()
        if curr != final:
            curr.color = WHITE

        if f:
            curr.color = GREEN

        pygame.display.update()


def algorithm_A_star(draw, grid, start, end):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0
    f_score = {spot: float("inf") for row in grid for spot in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    open_set_hash = {start}

    nodes_visited = 0

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)
        nodes_visited += 1

        if current == end:
            reconstruct_path(came_from, end, draw, end, start)
            return g_score[end], nodes_visited

        for neighbor in current.neighbors:
            if (
                neighbor == grid[current.row + 1][current.col]
                or neighbor == grid[current.row][current.col + 1]
                or neighbor == grid[current.row - 1][current.col]
                or neighbor == grid[current.row][current.col - 1]
            ):
                temp_g_score = g_score[current] + 1
            else:
                temp_g_score = g_score[current] + 1.4

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)

        draw()
    return float("inf"), nodes_visited


def algorithm_dijkstra(draw, grid, start, end):
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    cost_so_far = {spot: float("inf") for row in grid for spot in row}
    cost_so_far[start] = 0

    nodes_visited = 0

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[1]
        nodes_visited += 1

        if current == end:
            reconstruct_path(came_from, end, draw, end, start)
            return cost_so_far[end], nodes_visited

        for neighbor in current.neighbors:
            if (
                neighbor == grid[current.row + 1][current.col]
                or neighbor == grid[current.row][current.col + 1]
                or neighbor == grid[current.row - 1][current.col]
                or neighbor == grid[current.row][current.col - 1]
            ):
                new_cost = cost_so_far[current] + 1
            else:
                new_cost = cost_so_far[current] + 1.4

            if new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost
                open_set.put((priority, neighbor))
                came_from[neighbor] = current

        draw()
    return float("inf"), nodes_visited


def algorithm_bfs(draw, grid, start, end):
    queue = deque([start])
    came_from = {}
    visited = {spot: False for row in grid for spot in row}
    visited[start] = True

    nodes_visited = 0

    while queue:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = queue.popleft()
        nodes_visited += 1

        if current == end:
            reconstruct_path(came_from, end, draw, end, start)
            return nodes_visited  # BFS doesn't have a path cost, so we return just the nodes visited

        for neighbor in current.neighbors:
            if not visited[neighbor]:
                queue.append(neighbor)
                visited[neighbor] = True
                came_from[neighbor] = current

        draw()
    return nodes_visited


def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, rows)
            grid[i].append(spot)
    return grid


def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))


def draw(win, grid, rows, width):
    win.fill(WHITE)
    for row in grid:
        for spot in row:
            spot.draw(win)
    draw_grid(win, rows, width)
    pygame.display.update()


def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos
    row = y // gap
    col = x // gap
    return row, col


def main(win, width, targets):
    ROWS = 40
    grid = make_grid(ROWS, width)
    start = None
    ends = targets

    run = True
    algorithms = {
        "A*": algorithm_A_star,
        "Dijkstra": algorithm_dijkstra,
        "BFS": algorithm_bfs,
    }

    while run:
        draw(win, grid, ROWS, width)
        for row in range(40):
            for column in range(40):
                if (row in range(4, 17) or row in range(19, 36)) and (
                    column in range(4, 8)
                    or column in range(11, 15)
                    or column in range(18, 22)
                    or column in range(25, 29)
                    or column in range(32, 36)
                ):
                    grid[column][row].make_barrier()
                if (row == 0 or row == 39) and (
                    column in range(7, 12) or column in range(28, 33)
                ):
                    grid[column][row].make_rep()
                if (row == 0 and column in range(7, 12)) or (
                    row == 39 and column in range(28, 33)
                ):
                    grid[column][row].make_del()
                if (row in range(17, 23)) and column == 0:
                    grid[column][row].make_cs()

        # Set the start position
        if not start:
            start = grid[0][0]  # Starting position (can be adjusted)
            start.make_start()

        if start and ends:
            for target in ends:
                for row in grid:
                    for spot in row:
                        spot.update_neighbors(grid)
                target_row, target_col = target
                grid[target_row][target_col].make_target()
                draw(win, grid, ROWS, width)  # Draw the target

                # Dictionary to store the results for comparison
                results = {}
                print("Iteration:")
                for algo_name, algo_func in algorithms.items():
                    # Reset the grid before each run
                    for row in grid:
                        for spot in row:
                            if not spot.is_barrier():
                                spot.reset()
                    start.make_start()
                    grid[target_row][target_col].make_target()

                    start_time = time.time()
                    if algo_name == "BFS":
                        nodes_visited = algo_func(
                            lambda: draw(win, grid, ROWS, width),
                            grid,
                            start,
                            grid[target_row][target_col],
                        )
                        cost = "N/A"  # BFS does not provide a path cost
                    else:
                        cost, nodes_visited = algo_func(
                            lambda: draw(win, grid, ROWS, width),
                            grid,
                            start,
                            grid[target_row][target_col],
                        )
                    end_time = time.time()
                    duration = end_time - start_time

                    results[algo_name] = {
                        "Cost": cost,
                        "Nodes Visited": nodes_visited,
                        "Time (s)": duration,
                    }

                    # Print results for each algorithm
                    print(f"Algorithm: {algo_name}")
                    print(f"Path Cost: {cost}")
                    print(f"Nodes Visited: {nodes_visited}")
                    print(f"Time Taken: {duration:.4f} seconds")
                    print("")

                start = grid[target_row][target_col]
                start.make_start()

            ends = []
            run = False  # Exit the loop when all targets are visited

        draw(win, grid, ROWS, width)

    pygame.quit()
    print("All locations done")


# Get target locations from product codes
target_locations = [product_location_mapping[code] for code in product_codes]

# Call the main function with all target locations
main(WIN, WIDTH, target_locations)
