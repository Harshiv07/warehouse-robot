import pygame
import math
from queue import PriorityQueue
import time
import threading

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


def algorithm(draw, grid, start, end):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0
    f_score = {spot: float("inf") for row in grid for spot in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw, end, start)
            return True

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
    return False


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


def main(win, width):
    ROWS = 40
    grid = make_grid(ROWS, width)
    start = None
    ends = []

    run = True

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

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]:  # LEFT
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                if not start and spot != ends:
                    start = spot
                    start.make_start()

                elif len(ends) < 6 and spot != start:
                    ends.append(spot)
                    spot.make_end()

            elif pygame.mouse.get_pressed()[2]:  # RIGHT
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                spot.reset()
                if spot == start:
                    start = None
                elif spot in ends:
                    ends.remove(spot)

        if start and len(ends) == 6:
            for i in range(len(ends)):
                for row in grid:
                    for spot in row:
                        spot.update_neighbors(grid)

                if algorithm(
                    lambda: draw(win, grid, ROWS, width), grid, start, ends[i]
                ):
                    for row in grid:
                        for spot in row:
                            if spot.color == ROBOT:
                                spot.color = WHITE

            start = None
            ends = []

        draw(win, grid, ROWS, width)

    pygame.quit()


main(WIN, WIDTH)
