import numpy as np
import random
import math
import pandas as pd

input1 = np.array([[0, 6, 4, 5],
                   [1, 3, 3, 9],
                   [4, 9, 2, 1],
                   [9, 6, 1, 2],
                   [2, 3, 4, 5]])

input2 = np.array([[0, 6, 4, 5, 1, 4, 3, 5, 6, 8, 7],
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

input3 = np.array([[1, 9, 9, 9, 9, 9],
                   [1, 1, 9, 1, 1, 1],
                   [9, 1, 9, 1, 9, 1],
                   [9, 1, 9, 1, 9, 1],
                   [9, 1, 9, 1, 9, 1],
                   [9, 1, 9, 1, 9, 1],
                   [9, 1, 1, 1, 9, 1]])


def ant_colony_optimization(input):
    grid = input
    row = grid.shape[0]
    col = grid.shape[1]

    pheromone_grid = np.ones((row, col))
    gens = 50
    evap_rate = 0.5
    ants = 32
    # alpha = 6
    # shortest = [], 10000000000

    def calculate_shortest_path(grid, pheromone_grid, evap_rate, gens, ants):
        shortest_path_found = []
        for generation in range(gens):
            shortest_distance = []
            history = []
            interim_grid = np.zeros((row, col))
            print("GENERATION ", generation)
            for ant in range(ants):

                ant_history, distance_traveled = traverse_grid(
                    grid, pheromone_grid)

                shortest_distance.append(distance_traveled)

                # Stacks the positions traversed by all ants while dividing the quality of the position
                # by the total distance traveled by each ant. This is the formula from Wikipedia
                for (x, y) in ant_history:
                    interim_grid[x][y] += grid[x][y] / distance_traveled
                # if ant == ants - 1:
                #     print(distance_traveled)

            shortest_path_found.append(np.min(shortest_distance))

            pheromone_grid = (1 - evap_rate) * pheromone_grid + interim_grid

        print("shortest paths for each gen: ", shortest_path_found)

    def traverse_grid(grid, pheromone_grid):
        # print("NEW ant: ")
        current_node = (0, 0)
        end_node = (row - 1, col - 1)
        history = set()
        history.add(current_node)
        distance_traveled = 0
        potential_nodes = (calculate_potential_nodes(history, current_node))

        while (current_node != end_node) and potential_nodes != 0:
            # print("current node:", grid[current_node[0]]
            #       [current_node[1]], " pos: ", current_node)
            distance_traveled += grid[current_node[0]][current_node[1]]

            if (current_node[0], current_node[1] + 1) == end_node or (current_node[0] + 1, current_node[1]) == end_node:

                return history, distance_traveled

            # print(potential_nodes)
            potential_nodes_sum = 0
            # calculate sum of each possible node for the probability usage
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

            if chosen_prob == north_prob:
                current_node = (current_node[0] - 1, current_node[1])
            elif chosen_prob == south_prob:
                current_node = (current_node[0] + 1, current_node[1])
            elif chosen_prob == west_prob:
                current_node = (current_node[0], current_node[1] - 1)
            elif chosen_prob == east_prob:
                current_node = (current_node[0], current_node[1] + 1)

            history.add(current_node)

            potential_nodes = calculate_potential_nodes(history, current_node)

        return history, distance_traveled

    def calculate_potential_nodes(history, curr_node):
        a = set()
        if curr_node[1] < col - 1 and curr_node[1] > 0:
            if curr_node[0] - 1 >= 0 and (curr_node[0] - 1, curr_node[1]) not in history:

                a.add((curr_node[0] - 1, curr_node[1]))

        if curr_node[0] + 1 <= row - 1 and (curr_node[0] + 1, curr_node[1]) not in history:

            a.add((curr_node[0] + 1, curr_node[1]))

        if curr_node[0] < row - 1 and curr_node[0] > 0:
            if curr_node[1] - 1 >= 0 and (curr_node[0], curr_node[1] - 1) not in history:

                a.add((curr_node[0], curr_node[1] - 1))

        if curr_node[1] + 1 <= col - 1 and (curr_node[0], curr_node[1] + 1) not in history:

            a.add((curr_node[0], curr_node[1] + 1))

        return a

    def roullete_select(probabilities):
        # Sorting the probabilities in case there are any zero probabilities.
        # This happens when the current point has reached one or more walls
        # and has no potential node to calculate probability for, therefore the zero probabilities
        probabilities.sort()
        r = random.uniform(0, 1)
        sum_index = 0
        chosen_prob = 0

        # sum_index acts as the cummulative sum as it increments
        #  without any need for extra storage use
        for i in probabilities:
            if sum_index <= r:
                chosen_prob = i
            sum_index += i

        return chosen_prob

    def calculate_north(a):
        north = pheromone_grid[a[0] - 1][a[1]]*(1/grid[a[0] - 1][a[1]])
        return north

    def calculate_west(a):
        west = pheromone_grid[a[0]][a[1] - 1]*(1/grid[a[0]][a[1] - 1])
        return west

    def calculate_east(a):
        east = pheromone_grid[a[0]][a[1] + 1]*(1/grid[a[0]][a[1] + 1])
        return east

    def calculate_south(a):
        south = pheromone_grid[a[0] + 1][a[1]]*(1/grid[a[0] + 1][a[1]])
        return south

    calculate_shortest_path(grid, pheromone_grid, evap_rate, gens, ants)


# def djikstra(input):
#     grid = input


ant_colony_optimization(input1)
