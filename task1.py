import sys
import os
import numpy as np
import random
import math
import pandas as pd

input1 = np.array([[0, 6, 4, 5],
                   [1, 3, 3, 9],
                   [4, 9, 2, 1],
                   [9, 6, 1, 2],
                   [2, 3, 4, 5]])

input2_v1 = np.array([[1, 9, 9, 9, 9, 9],
                      [1, 1, 0, 1, 0, 1],
                      [9, 1, 9, 1, 9, 1],
                      [9, 1, 9, 1, 9, 1],
                      [9, 1, 9, 1, 9, 1],
                      [9, 1, 9, 1, 9, 1],
                      [9, 1, 1, 1, 9, 1]])

input2_v2 = np.array([[1, 9, 9, 9, 9, 9],
                      [1, 1, 1, 1, 1, 1],
                      [9, 1, 9, 1, 9, 1],
                      [9, 1, 9, 1, 9, 1],
                      [9, 1, 9, 1, 9, 1],
                      [9, 1, 9, 1, 9, 1],
                      [9, 1, 1, 1, 9, 1]])

input3 = np.array([[1, 6, 4, 5, 1, 4, 3, 6, 8, 7],
                   [1, 3, 3, 9, 1, 4, 3, 6, 2, 1],
                   [4, 1, 9, 1, 1, 4, 3, 6, 5, 3],
                   [9, 6, 1, 2, 1, 4, 3, 6, 2, 1],
                   [1, 3, 5, 4, 1, 4, 3, 6, 8, 4],
                   [8, 7, 2, 9, 1, 4, 3, 6, 7, 5],
                   [8, 7, 2, 9, 1, 4, 3, 6, 7, 5],
                   [1, 6, 3, 5, 1, 4, 3, 6, 2, 2],
                   [8, 7, 2, 9, 1, 4, 3, 6, 7, 5],
                   [1, 6, 3, 5, 1, 4, 3, 6, 2, 2]])

input4 = np.array([[0, 6, 4, 5, 1, 4, 3, 5, 6, 8, 7],
                   [1, 3, 3, 9, 1, 4, 3, 5, 6, 2, 1],
                   [4, 1, 9, 1, 1, 4, 3, 5, 6, 5, 3],
                   [9, 6, 1, 2, 1, 4, 3, 5, 6, 2, 1],
                   [1, 3, 5, 4, 1, 4, 3, 5, 6, 8, 4],
                   [8, 7, 2, 9, 1, 4, 3, 5, 6, 7, 5],
                   [1, 6, 3, 5, 1, 4, 3, 5, 6, 2, 2],
                   [8, 7, 2, 9, 1, 4, 3, 5, 6, 7, 5],
                   [1, 6, 3, 5, 1, 4, 3, 5, 6, 2, 2],
                   [8, 7, 2, 9, 1, 4, 3, 5, 6, 7, 5],
                   [1, 6, 3, 5, 1, 4, 3, 5, 6, 2, 2]])


def ant_colony_optimization(input):
    grid = input
    row = grid.shape[0]
    col = grid.shape[1]

    pheromone_grid = np.ones((row, col))
    gens = 40
    evap_rate = 0.7
    ants = 1200
    alpha = 1.5

    def calculate_shortest_path(grid, pheromone_grid, evap_rate, gens, ants):
        shortest_path_found = []
        for generation in range(gens):
            ants_distances = []
            history = []
            interim_grid = np.zeros((row, col))
            pheromone_grid = (1 - evap_rate) * pheromone_grid + interim_grid
            # print("GENERATION ", generation)
            for ant in range(ants):

                ant_history, distance_traveled = traverse_grid(
                    grid, pheromone_grid)

                ants_distances.append(distance_traveled)
                if distance_traveled == 0:
                    distance_traveled = 0.7
                # Stacks the positions traversed by all ants while dividing the alpha constant
                # by the total distance traveled by each ant. This is the formula from Wikipedia
                for (x, y) in ant_history:

                    interim_grid[x][y] += alpha / distance_traveled
                # if ant == ants - 1:
                #     print(distance_traveled)

            shortest_path_found.append(np.min(ants_distances))

            # pheromone_grid = (1 - evap_rate) * pheromone_grid + interim_grid

        print("shortest path: ", np.min(
            shortest_path_found + grid[row - 1][col - 1] - grid[0][0]))

    def traverse_grid(grid, pheromone_grid):
        current_node = (0, 0)
        end_node = (row - 1, col - 1)
        history = set()
        history.add(current_node)
        distance_traveled = 0
        potential_nodes = (calculate_potential_nodes(history, current_node))

        while (current_node != end_node) and len(potential_nodes) != 0:
            distance_traveled += grid[current_node[0]][current_node[1]]

            current_node = choose_node(current_node, potential_nodes)

            history.add(current_node)

            potential_nodes = calculate_potential_nodes(history, current_node)

        if current_node != end_node:
            distance_traveled = 100000

        return history, distance_traveled

    def choose_node(current_node, potential_nodes):
        potential_nodes_sum = 0
        # calculate sum of each possible node for the probability later
        for (x, y) in potential_nodes:
            if x == current_node[0] + 1:
                potential_nodes_sum += calculate_south(current_node)
            elif x == current_node[0] - 1:
                potential_nodes_sum += calculate_north(current_node)
            elif y == current_node[1] + 1:
                potential_nodes_sum += calculate_east(current_node)
            elif y == current_node[1] - 1:
                potential_nodes_sum += calculate_west(current_node)

        north_prob = 0
        west_prob = 0
        east_prob = 0
        south_prob = 0
        # calculate probabilities
        for (x, y) in potential_nodes:
            if x == current_node[0] + 1:
                south_prob = calculate_south(
                    current_node)/potential_nodes_sum
            if x == current_node[0] - 1:
                north_prob = calculate_north(
                    current_node)/potential_nodes_sum
            if y == current_node[1] + 1:
                east_prob = calculate_east(
                    current_node)/potential_nodes_sum
            if y == current_node[1] - 1:
                west_prob = calculate_west(
                    current_node)/potential_nodes_sum

        chosen_prob = roullete_select(
            [north_prob, west_prob, east_prob, south_prob])
        # roullete_select returns the index of the chosen probability
        # in the order which they we're inputted above.
        # For index 0 it's north_prob, for index 1 it's west_prob and so on... .
        # This 'if' function distinguishes which potential node to return based on that index rule.
        if chosen_prob == 0:
            return (current_node[0] - 1, current_node[1])
        elif chosen_prob == 1:
            return (current_node[0], current_node[1] - 1)
        elif chosen_prob == 2:
            return (current_node[0], current_node[1] + 1)
        elif chosen_prob == 3:
            return (current_node[0] + 1, current_node[1])

    def calculate_potential_nodes(history, curr_node):
        a = set()

        if curr_node[0] + 1 <= row - 1 and (curr_node[0] + 1, curr_node[1]) not in history:

            a.add((curr_node[0] + 1, curr_node[1]))

        # if curr_node[1] < col - 1 and curr_node[1] > 0:
        if curr_node[0] - 1 >= 0 and (curr_node[0] - 1, curr_node[1]) not in history:

            a.add((curr_node[0] - 1, curr_node[1]))

        # if curr_node[0] < row - 1 and curr_node[0] > 0:
        if curr_node[1] - 1 >= 0 and (curr_node[0], curr_node[1] - 1) not in history:

            a.add((curr_node[0], curr_node[1] - 1))

        if curr_node[1] + 1 <= col - 1 and (curr_node[0], curr_node[1] + 1) not in history:

            a.add((curr_node[0], curr_node[1] + 1))

        return a

    def roullete_select(probabilities):
        r = random.uniform(0, 1)
        sum_index = 0
        # chosen_prob = 0
        chosen_prob_index = 0
        # sum_index acts as the cummulative sum as it increments
        #  without any need for extra storage use
        for i in range(len(probabilities)):
            # Skipping the probabilities in case there are any zero probabilities.
            # This happens when the current point has reached one or more walls
            # and has no potential node to calculate probability for.
            if probabilities[i] == 0:
                continue

            if sum_index <= r:
                chosen_prob_index = i

            sum_index += probabilities[i]

        return chosen_prob_index

    def calculate_north(a):
        if grid[a[0] - 1][a[1]] == 0:
            # return pheromone_grid[a[0] - 1][a[1]]
            return 0.49
        return pheromone_grid[a[0] - 1][a[1]]*(1/grid[a[0] - 1][a[1]])

    def calculate_west(a):
        if grid[a[0]][a[1] - 1] == 0:
            # return pheromone_grid[a[0]][a[1] - 1]
            return 0.49
        return pheromone_grid[a[0]][a[1] - 1]*(1/grid[a[0]][a[1] - 1])

    def calculate_east(a):
        if grid[a[0]][a[1] + 1] == 0:
            # return pheromone_grid[a[0]][a[1] + 1]
            return 0.49
        east = pheromone_grid[a[0]][a[1] + 1]*(1/grid[a[0]][a[1] + 1])
        return east

    def calculate_south(a):
        if grid[a[0] + 1][a[1]] == 0:
            # return pheromone_grid[a[0] + 1][a[1]]
            return 0.49
        return pheromone_grid[a[0] + 1][a[1]]*(1/grid[a[0] + 1][a[1]])

    calculate_shortest_path(grid, pheromone_grid, evap_rate, gens, ants)


def heuristic(input):
    # np.random.seed(0)
    size_x = 5
    size_y = 5
    random_matrix = input

    time_spent = random_matrix[0, 0]

    x = y = 0

    while x < size_x and y < size_y:
        print((x, y), random_matrix[x, y])

        if x < size_x and y < size_y:
            look_right = random_matrix[x, y + 1]
            look_down = random_matrix[x + 1, y]

            if look_right == min(look_right, look_down):
                time_spent += look_right
                y += 1
            else:
                time_spent += look_down
                x += 1

        if size_x == x and y < size_y:
            time_spent += random_matrix[x, y + 1]
            y += 1

        if size_y == y and x < size_x:
            time_spent += random_matrix[x + 1, y]
            x += 1

    print((x, y), random_matrix[x, y])
    print(time_spent)
    print(random_matrix)


def djikstra(input):
    start = (0, 0)


rand_input = np.random.randint(10, size=(5, 5))
# print(rand_input)
ant_colony_optimization(input4)
