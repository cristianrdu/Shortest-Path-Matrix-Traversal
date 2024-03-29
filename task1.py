import sys
import os
import numpy as np
import random
import math
import pandas as pd
import time

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

# Game class


class Task_one_game():
    # Initializes the game by storing the input grid to be used on the different path finding modes.
    def __init__(self, input):
        self.input = input

    # Reference: YouTube. Oct. 4, 2018.
    # How the Ant Colony Optimization algorithm works - YouTube.
    # [ONLINE] Available at: https://www.youtube.com/watch?v=783ZtAF4j5g.
    # [Accessed 19 December 2020].
    # Reference: Wikipedia.
    # Ant colony optimization algorithms - Wikipedia.
    # [ONLINE] Available at: https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms#Algorithm_and_formulae .
    # [Accessed 19 December 2020].
    def ant_colony_optimization(self, generations, ants):
        grid = self.input
        row = grid.shape[0]
        col = grid.shape[1]

        # Initializes the pheromone grid with ones
        pheromone_grid = np.ones((row, col))
        gens = generations
        # Evaporation rate for the formula
        # If a point in the grid doesn't get traversed, its pheromone gradually disipates by the rate
        evap_rate = 0.7
        ants = ants
        alpha = 1.5

        def calculate_shortest_path(grid, pheromone_grid, evap_rate, gens, ants):
            # Stores the shortest distances for each generation
            shortest_dist_found = []
            # Iterates through each generation
            for gen in range(gens):
                # Stores the distances of each ant for the current generation
                ants_distances = []
                interim_grid = np.zeros((row, col))
                pheromone_grid = (1 - evap_rate) * \
                    pheromone_grid + interim_grid
                # print("GENERATION ", generation)
                for ant in range(ants):
                    # For each ant, traverses the grid and gets its path/history and distance made
                    ant_history, distance_traveled = traverse_grid(
                        grid, pheromone_grid)

                    ants_distances.append(distance_traveled)
                    # If the ant takes 0 distance to reach its destination,
                    # then it updates the interim_grid by the alpha constant
                    if distance_traveled == 0:
                        distance_traveled = 0.7
                    # Stacks the positions traversed of each ant while dividing the alpha constant
                    # by the total distance traveled by each ant. This is the formula from Wikipedia
                    for (x, y) in ant_history:
                        interim_grid[x][y] += alpha / distance_traveled
                # appends the shortest distance after the generation is over
                shortest_dist_found.append(np.min(ants_distances))
                # Updates the pheromones with the interim_grid by using the formula from Wikipedia.
                pheromone_grid = (1 - evap_rate) * \
                    pheromone_grid + interim_grid
            # Removes the starting point added in by the ant to the total,
            # adds in the final destination point, as it was not added in the algorithm,
            # and prints out the shortest path
            print("ACO shortest path: ", np.min(
                shortest_dist_found + grid[row - 1][col - 1] - grid[0][0]))

        def traverse_grid(grid, pheromone_grid):
            # Starts at 0,0
            current_node = (0, 0)
            end_node = (row - 1, col - 1)
            # Initializes the points set traversed by the ant with the starting point
            history = set()
            history.add(current_node)
            distance_traveled = 0
            # Generates the initial potential nodes (1,0) and (0,1)
            potential_nodes = (
                calculate_potential_nodes(history, current_node))
            # Parses through the grid while it hasn't reached the end point
            # or until there still are potential nodes(nodes act as points on the grid) to go to
            while (current_node != end_node) and len(potential_nodes) != 0:
                # Updates distance traveled with the current node's value
                distance_traveled += grid[current_node[0]][current_node[1]]
                # Updates the current node
                current_node = choose_node(current_node, potential_nodes)
                # Adds it to the history
                history.add(current_node)
                # Generates new nodes to choose from for the next iteration
                potential_nodes = calculate_potential_nodes(
                    history, current_node)
            # Sets the distance_traversed to 100000 if the ant hasn't reached the end
            if current_node != end_node:
                distance_traveled = 100000

            return history, distance_traveled

        # The formulas used in this method are from the References above.
        def choose_node(current_node, potential_nodes):
            potential_nodes_sum = 0
            # Calculates sum of each possible node for use in the main probability
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
            # Calculates the probabilities
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
            # roullete_select takes into acocunt chances of probabilities being 0 when passed as parameters
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
            # Calculates if the nodes from the north/east/west/south region with respect to the current_node
            # are eligible to be added in the potential_nodes set for the next iteration
            # by checking if it hits any walls as well as the ant's history
            if curr_node[0] + 1 <= row - 1 and (curr_node[0] + 1, curr_node[1]) not in history:

                a.add((curr_node[0] + 1, curr_node[1]))

            if curr_node[0] - 1 >= 0 and (curr_node[0] - 1, curr_node[1]) not in history:

                a.add((curr_node[0] - 1, curr_node[1]))

            if curr_node[1] - 1 >= 0 and (curr_node[0], curr_node[1] - 1) not in history:

                a.add((curr_node[0], curr_node[1] - 1))

            if curr_node[1] + 1 <= col - 1 and (curr_node[0], curr_node[1] + 1) not in history:

                a.add((curr_node[0], curr_node[1] + 1))

            return a

        def roullete_select(probabilities):
            # Randomizes a number to be used in the probability loop below
            r = random.uniform(0, 1)
            # sum_index acts as the cummulative sum as it increments
            #  without any need for extra storage use
            sum_index = 0
            chosen_prob_index = 0
            # Iterates through the number of probabilities
            for i in range(len(probabilities)):
                # Skipping the probabilities in case there are any zero probabilities.
                # This happens when the current point has reached one or more walls
                # and has no potential node to calculate probability for.
                if probabilities[i] == 0:
                    continue
                # If the random number is less than the incremental sum_index,
                # then it chooses the current index
                # For instance, if we have the probabilities:
                # [0.1,0.2,0.1,0.6] and random number 0.3
                # then it would choose the 2nd probability as 0.1 + 0.2 is 0.3, which is <= than the rand no.,
                # but if you add the next probability it reaches 0.4 which doesn't satisfy the condition,
                # making the chosen probability index to stay at 2, which is the 0.2 probability.
                if sum_index <= r:
                    chosen_prob_index = i

                sum_index += probabilities[i]

            return chosen_prob_index

        def calculate_north(a):
            # Calculates the probability for the north node, given from the formulas in the references above
            if grid[a[0] - 1][a[1]] == 0:
                return 0.49
            return pheromone_grid[a[0] - 1][a[1]]*(1/grid[a[0] - 1][a[1]])

        def calculate_west(a):
            # Calculates the probability for the west node, given from the formulas in the references above
            if grid[a[0]][a[1] - 1] == 0:
                return 0.49
            return pheromone_grid[a[0]][a[1] - 1]*(1/grid[a[0]][a[1] - 1])

        def calculate_east(a):
            # Calculates the probability for the east node, given from the formulas in the references above
            if grid[a[0]][a[1] + 1] == 0:
                return 0.49
            east = pheromone_grid[a[0]][a[1] + 1]*(1/grid[a[0]][a[1] + 1])
            return east

        def calculate_south(a):
            # Calculates the probability for the south node, given from the formulas in the references above
            if grid[a[0] + 1][a[1]] == 0:
                return 0.49
            return pheromone_grid[a[0] + 1][a[1]]*(1/grid[a[0] + 1][a[1]])

        calculate_shortest_path(grid, pheromone_grid, evap_rate, gens, ants)

    def heuristic(self):
        # Sets the boundaries for the while loop below
        size_x = self.input.shape[0] - 1
        size_y = self.input.shape[1] - 1
        grid = self.input

        time_spent = 0
        # Initializes the positions on the start
        x = y = 0
        while x <= size_x and y <= size_y:
            # Calculates the next point if it's not close to any wall
            if x < size_x and y < size_y:
                look_right = grid[x, y + 1]
                look_down = grid[x + 1, y]
                # Chooses the shortest distance point
                if look_right == min(look_right, look_down):
                    time_spent += look_right
                    y += 1
                else:
                    time_spent += look_down
                    x += 1
            # If it's next to the horizontal walls, it only goes to the right
            if size_x == x and y < size_y:
                time_spent += grid[x, y + 1]
                y += 1
            # If it's next to the vertical walls, it only goes below
            if size_y == y and x < size_x:
                time_spent += grid[x + 1, y]
                x += 1
            # If it has reached the end, break the loop
            if size_y == y and x == size_x:
                break

        print("Heuristic shortest path: ", time_spent)

    # Reference: Wikipedia.
    # Dijkstra's algorithm - Wikipedia.
    # [ONLINE] Available at: https://en.wikipedia.org/wiki/Dijkstra's_algorithm?fbclid=IwAR3EvRxdGdemWFGdZYVbyARZmViMWMtaoS18Ck4m7QYDVN22tCdl95WmNOk#Algorithm .
    # [Accessed 19 December 2020].
    def dijkstra(self):

        grid = self.input

        row = grid.shape[0]
        col = grid.shape[1]
        # Initializes the node params
        start_node = (0, 0)
        end_node = (row - 1, col - 1)
        current_node = start_node
        # Sets the distance grid to a high number (easier to use/visualise compared to infinity - wikipedia)
        distance_grid = np.full((row, col), 99999)
        # Sets the start distance to 0
        distance_grid[0][0] = 0
        # Creates a set of unvisitied nodes and adds all the nodes to it - marked as unvisited
        unvisited_nodes = set()
        for x in range(row):
            for y in range(col):
                unvisited_nodes.add((x, y))

        def start(current_node):
            # First neighbour nodes are the right and bottom ones
            neighbour_nodes = ((1, 0), (0, 1))

            # Parses through the grid until it visits end_node
            while end_node in unvisited_nodes:
                # Updates the distances of the neighbour nodes
                update_distances(current_node, neighbour_nodes)
                # Marks the current node as visited
                unvisited_nodes.remove(current_node)

                current_node = get_next_node()
                # Gets the neighbours of the new current_node
                neighbour_nodes = update_neighbour_nodes(current_node)

            print("Dijkstra shortest path: ",
                  distance_grid[end_node[0]][end_node[1]])

        def update_distances(curr_node, neighbour_nodes):
            for (x, y) in neighbour_nodes:
                distance = distance_grid[curr_node[0]
                                         ][curr_node[1]] + grid[x][y]
                if distance_grid[x][y] > distance:
                    distance_grid[x][y] = distance

        def get_next_node():
            # Sets an initial min distance to be compared to
            min_distance = 9999999
            # Initialises min_node with a random pos tuple
            min_node = (999, 999)
            # Searches through the unvisited nodes the smallest distance and updates min_node
            for (x, y) in unvisited_nodes:
                if min_distance > distance_grid[x][y]:
                    min_distance = distance_grid[x][y]
                    min_node = (x, y)

            return min_node

        def update_neighbour_nodes(curr_node):
            updated_nodes = set()
            x = curr_node[0]
            y = curr_node[1]
            # Checks for walls or possible neighbours that haven't been visited
            # based on the indices of the curr_node
            if x + 1 < row and (x+1, y) in unvisited_nodes:
                updated_nodes.add((x+1, y))
            if x - 1 >= 0 and (x-1, y) in unvisited_nodes:
                updated_nodes.add((x-1, y))
            if y + 1 < col and (x, y + 1) in unvisited_nodes:
                updated_nodes.add((x, y + 1))
            if y - 1 >= 0 and (x, y - 1) in unvisited_nodes:
                updated_nodes.add((x, y - 1))

            return updated_nodes

        start(current_node)


rand_input = np.random.randint(10, size=(11, 11))

main_input = input4
game = Task_one_game(main_input)
print(main_input)

start = time.time()
game.heuristic()
end = time.time()
print("Time elapsed for Heuristic: ", end - start)

start = time.time()
game.dijkstra()
end = time.time()
print("Time elapsed for Dijkstra: ", end - start)

start = time.time()
game.ant_colony_optimization(generations=50, ants=800)
end = time.time()
print("Time elapsed for ACO: ", end - start)
