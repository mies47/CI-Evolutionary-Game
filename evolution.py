from random import random, randint
from player import Player
import numpy as np
from config import CONFIG
import copy

class Evolution():

    def __init__(self, mode):
        self.mode = mode

    # calculate fitness of players
    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]

    def mutate(self, child):

        # TODO
        # child: an object of class `Player`
        sigma = 0.5
        mu = 0
        pm = 0.5
        if random() <= pm:
            child.nn.w_in_hidden += np.random.normal(mu, sigma, size=child.nn.w_in_hidden.shape)
        if random() <= pm:
            child.nn.w_hidden_out += np.random.normal(mu, sigma, size=child.nn.w_hidden_out.shape)
        # if random() <= pm:
        #     child.nn.w_hidden_hidden += np.random.normal(mu, sigma, size=child.nn.w_hidden_hidden.shape)
        if random() <= pm:
            child.nn.b_hidden += np.random.normal(mu, sigma, size=child.nn.b_hidden.shape)
        # if random() <= pm:
        #     child.nn.b_hidden1 += np.random.normal(mu, sigma, size=child.nn.b_hidden1.shape)
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
            # new_players = [copy.deepcopy(player) for player in prev_players[: num_players]]
            # for player in new_players:
            #     self.mutate(player)
            # TODO (additional): a selection method other than `fitness proportionate`

            def parent_selection(players, players_num=2):
                players_length = len(players)
                Q = 3
                best_match = -1
                best_match_index = 0
                selected = []
                for i in range(players_num):
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
            def do_crossover(player1, player2):
                new_player1 = copy.deepcopy(player1)
                new_player2 = copy.deepcopy(player2)
                w_in_hidden_crossover_point = randint(0, len(new_player1.nn.w_in_hidden))
                w_hidden_out_crossover_point = randint(0, len(new_player1.nn.w_hidden_out))
                b_hidden_crossover_point = randint(0, len(new_player1.nn.b_hidden))
                b_output_crossover_point = randint(0, len(new_player1.nn.b_output))

                new_player1.nn.w_in_hidden = np.concatenate(
                    (new_player1.nn.w_in_hidden[:w_in_hidden_crossover_point],
                    copy.deepcopy(new_player2.nn.w_in_hidden[w_in_hidden_crossover_point:])), axis=0)

                new_player2.nn.w_in_hidden = np.concatenate(
                    (new_player2.nn.w_in_hidden[:w_in_hidden_crossover_point],
                    copy.deepcopy(player1.nn.w_in_hidden[w_in_hidden_crossover_point:])), axis=0)

                new_player1.nn.w_hidden_out = np.concatenate(
                    (new_player1.nn.w_hidden_out[:w_hidden_out_crossover_point],
                    copy.deepcopy(new_player2.nn.w_hidden_out[w_hidden_out_crossover_point:])), axis=0)

                new_player2.nn.w_hidden_out = np.concatenate(
                    (new_player2.nn.w_hidden_out[:w_hidden_out_crossover_point],
                    copy.deepcopy(player1.nn.w_hidden_out[w_hidden_out_crossover_point:])), axis=0)

                new_player1.nn.b_hidden = np.concatenate(
                    (new_player1.nn.b_hidden[:b_hidden_crossover_point],
                    copy.deepcopy(new_player2.nn.b_hidden[b_hidden_crossover_point:])), axis=0)

                new_player2.nn.b_hidden = np.concatenate(
                    (new_player2.nn.b_hidden[:b_hidden_crossover_point],
                    copy.deepcopy(player1.nn.b_hidden[b_hidden_crossover_point:])), axis=0)

                new_player1.nn.b_output = np.concatenate(
                    (new_player1.nn.b_output[:b_output_crossover_point],
                    copy.deepcopy(new_player2.nn.b_output[b_output_crossover_point:])), axis=0)

                new_player2.nn.b_output = np.concatenate(
                    (new_player2.nn.b_output[:b_output_crossover_point],
                    copy.deepcopy(player1.nn.b_output[b_output_crossover_point:])), axis=0)

                return new_player1, new_player2

            parent_selection_num = int(num_players / 2)
            new_players = []
            for _ in range(parent_selection_num):
                selected = parent_selection(prev_players)
                new_player1, new_player2 = do_crossover(selected[0], selected[1])
                self.mutate(new_player1)
                self.mutate(new_player2)
                new_players.append(new_player1)
                new_players.append(new_player2)
            
            if 2 * parent_selection_num != num_players:
                selected = parent_selection(prev_players)
                new_player1, new_player2 = do_crossover(selected[0], selected[1])
                self.mutate(new_player1)
                new_players.append(new_player1)

            return new_players

    def next_population_selection(self, players, num_players):

        # TODO
        # num_players example: 100
        # players: an array of `Player` objects

        #Main task
        # players.sort(key=lambda player: player.fitness, reverse=True)
        # return players[: num_players]

        # TODO (additional): a selection method other than `top-k`
        #Q_tournament
        players_length = len(players)
        Q = 3
        best_match = -1
        best_match_index = 0
        selected_population = []
        for _ in range(num_players):
            for i in range(Q):
                random_index = randint(0, players_length - 1)
                if players[random_index].fitness > best_match:
                    best_match_index = random_index
                    best_match = players[random_index].fitness
            selected_population.append(copy.deepcopy(players[best_match_index]))
            best_match = -1
            best_match_index = 0

        return selected_population
        # TODO (additional): plotting        