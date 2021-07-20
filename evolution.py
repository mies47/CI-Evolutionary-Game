from random import random, randint
from player import Player
import numpy as np
from config import CONFIG
import copy
import json

class Evolution():

    def __init__(self, mode):
        self.mode = mode
        self.generation = 1
        self.successfull_mutation = 0
        self.total_pop = 0
        self.last_ps = 0
        self.converged = False

    # calculate fitness of players
    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]

    def mutate(self, child):

        # TODO
        # child: an object of class `Player`
        c = 0.9
        sigma = 0.9
        if self.last_ps > 0.2:
            sigma = sigma / c
        elif self.last_ps < 0.2:
            sigma = sigma * c

        mu = 0
        # pm = 0.6 if not self.converged else 1
        pm = 1
        if random() <= pm:
            child.nn.w_in_hidden += np.random.normal(mu, sigma, size=child.nn.w_in_hidden.shape)
        if random() <= pm:
            child.nn.w_hidden_out += np.random.normal(mu, sigma, size=child.nn.w_hidden_out.shape)
        if random() <= pm:
            child.nn.b_hidden += np.random.normal(mu, sigma, size=child.nn.b_hidden.shape)
        if random() <= pm:
            child.nn.b_output += np.random.normal(mu, sigma, size=child.nn.b_output.shape)

    def generate_new_population(self, num_players, prev_players=None):

        # in first generation, we create random players
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]

        else:

            # TODO
            # num_players example: 150
            # prev_players: an array of `Player` objects

            #Main task

            # prev_players.sort(key=lambda player: player.fitness, reverse=True)
            # for player in prev_players[: num_players]:
            #   self.mutate(player)
            #   new_player = copy.deepcopy(player)
            # return new_player

            # new_players = []

            def roulette(items, n):
                total = float(sum(w.fitness for w in items))
                i = 0
                w, v = items[0].fitness, items[0]
                while n:
                    x = total * (1 - np.random.random() ** (1.0 / n))
                    total -= x
                    while x > w:
                        x -= w
                        i += 1
                        w, v = items[i].fitness, items[i]
                    w -= x
                    yield v
                    n -= 1


            # TODO (additional): a selection method other than `fitness proportionate`

            def parent_selection(players, players_num=2):
                players_length = len(players)
                Q = 1
                best_match = -1
                best_match_index = 0
                selected = []
                for _ in range(players_num):
                    # Q tournament selection
                    for _ in range(Q):
                        random_index = randint(0, players_length - 1)
                        if players[random_index].fitness > best_match:
                            best_match_index = random_index
                            best_match = players[random_index].fitness
                    
                    selected.append(players[best_match_index])
                    best_match = -1
                    best_match_index = 0

                return selected

            # TODO (additional): implementing crossover
            def second_crossover(players):
                crossover_operation = randint(1, 4)
                nn_attr = ['w_in_hidden', 'w_hidden_out', 'b_hidden', 'b_output']

                if crossover_operation == 1: #local intermediary
                    selected = parent_selection(players)
                    player1, player2 = copy.deepcopy(selected[0]), selected[1]

                    for attr in nn_attr:
                        player_attr = getattr(player1.nn, attr)
                        player_attr = (getattr(player1.nn, attr) + getattr(player2.nn, attr))/ 2


                elif crossover_operation == 2: #global intermediary
                    for attr in nn_attr:
                        for i in range(len(getattr(players[0].nn, attr))):
                            selected = parent_selection(players)
                            player1, player2 = copy.deepcopy(selected[0]), selected[1]
                            player_attr = getattr(player1.nn, attr)
                            player_attr[i] = (getattr(player1.nn, attr)[i] + getattr(player2.nn, attr)[i])/ 2


                elif crossover_operation == 3: #local discrete
                    selected = parent_selection(players)
                    player1, player2 = copy.deepcopy(selected[0]), selected[1]

                    for attr in nn_attr:
                        if randint(0, 1) == 1:
                            player_attr = getattr(player1.nn, attr)
                            player_attr = copy.deepcopy(getattr(player2.nn, attr))

                else: # global discrete
                    selected = parent_selection(players)
                    player1, player2 = copy.deepcopy(selected[0]), selected[1]

                    for attr in nn_attr:
                        for i in range(len(getattr(players[0].nn, attr))):
                            if randint(0, 1) == 1:
                                player_attr = getattr(player1.nn, attr)
                                player_attr[i] = copy.deepcopy(getattr(player2.nn, attr)[i])

                return player1
                
            new_players = []

            fitness_arr = np.array([player.fitness for player in prev_players]) 

            # avg_fitness = np.mean(fitness_arr)
            max_fitness = np.max(fitness_arr)

            for player in prev_players:
                if player.fitness == max_fitness:
                    self.successfull_mutation += 1

            if self.generation % 2 == 0:
                self.last_ps = self.successfull_mutation / self.total_pop
                self.total_pop = 0
                self.successfull_mutation = 0


            # if avg_fitness >= 0.4 * max_fitness:
            #     self.converged = True
            # else:
            #     self.converged = False

            pc = 0.4
            for _ in range(num_players):
                if random() < pc:
                    child = second_crossover(prev_players)
                else:
                    child = copy.deepcopy(parent_selection(prev_players, 1)[0])
                self.mutate(child)
                new_players.append(child)

            return new_players
            

    def next_population_selection(self, players, num_players):

        # TODO
        # num_players example: 100
        # players: an array of `Player` objects

        self.generation += 1
        self.total_pop += num_players
        fitness_arr = np.array([player.fitness for player in players])

        #Main task
        # players.sort(key=lambda player: player.fitness, reverse=True)
        # return players[: num_players]

        # TODO (additional): a selection method other than `top-k`
        #Q_tournament
        new_players = copy.deepcopy(players)
        players_length = len(new_players)
        Q = 5
        best_match = -1
        best_match_index = 0
        selected_population = []
        for _ in range(num_players):
            for i in range(Q):
                random_index = randint(0, players_length - 1)
                if new_players[random_index].fitness > best_match:
                    best_match_index = random_index
                    best_match = new_players[random_index].fitness
            selected_population.append(copy.deepcopy(new_players[best_match_index]))
            best_match = -1
            best_match_index = 0
            new_players.pop(best_match_index)
            players_length -= 1

        

        # TODO (additional): plotting
        with open('plotting.json', 'r') as f:
            data = json.load(f)
            if not data:
                data = {
                    'avg': [],
                    'min': [],
                    'max': []
                }
            
            mean = int(np.mean(fitness_arr))
            maximum = int(np.max(fitness_arr))
            minimum = int(np.min(fitness_arr))
            gen_num = len(data['avg']) + 1
            data['avg'].append((gen_num, mean))
            data['min'].append((gen_num, minimum))
            data['max'].append((gen_num, maximum))

        with open('plotting.json', 'w') as f:
            json.dump(data, f)

        return selected_population