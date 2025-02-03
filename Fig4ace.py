import math
import random

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter

num_nodes = 3000
num_iterations = 1000

heatmap1 = []
heatmap2 = []
heatmap3 = []


for c in np.arange(0, 5.25, 0.25):
    row1 = []
    row2 = []
    row3 = []
    for b0 in np.arange(0, 5.25, 0.25):
        DELTA = 1
        b1 = b0 + DELTA
        b2 = b1 + DELTA



        Game_State = [0, 1, 2]
        Action = [1, 0]








        kappa = 0.1
        gamma = 0

        G = nx.watts_strogatz_graph(num_nodes, k=6, p=0.4)


        Actions = np.random.choice([0, 1], size=num_nodes)
        policy = [np.random.choice([0, 1], size=int(1 / (1-gamma))) for _ in range(num_nodes)]


        Current_GameState = [random.choice([0, 1, 2]) for _ in range(num_nodes)]



        def reward(node, state, action):
            neighbors = list(G.neighbors(node))

            if state == 0:
                R, S, T, P = b0-c, -c, b0, 0
            elif state == 1:
                R, S, T, P = b1-c, -c, b1, 0
            elif state == 2:
                R, S, T, P = b2-c, -c, b2, 0

            reward = 0
            i = 0
            for neighbor in neighbors:

                if action == 1 and Actions[neighbor] == 1:
                    reward += R
                elif action == 1 and Actions[neighbor] == 0:
                    reward += S
                elif action == 0 and Actions[neighbor] == 1:
                    reward += T
                else:
                    reward += P
                i+=1

            return reward



        def get_best_policy(node_in):
            n = int(1 / (1-gamma))

            Guess_actions = Actions.copy()
            Guess_State = Current_GameState.copy()
            best_note = node_in
            best_policy = []

            for i in range(n):
                best_action = Guess_actions[node_in]
                best_benefit = reward(node_in,Guess_State[node_in],Guess_actions[node_in])

                for neighbor in G.neighbors(node_in):
                    if reward(neighbor,Guess_State[neighbor],Guess_actions[neighbor]) > best_benefit:
                        best_note = neighbor
                        best_action = Guess_actions[neighbor]
                        best_benefit = reward(neighbor,Guess_State[neighbor],Guess_actions[neighbor])


                probability_to_switch = 1 / (1 + (math.exp((reward(node_in, Guess_State[node_in],Guess_actions[node_in]) - reward(best_note,Guess_State[best_note],Guess_actions[best_note])) / (kappa * (1 / (1 - (i * gamma)))))))


                if np.random.rand() < probability_to_switch:
                    Guess_actions[node_in] = best_action


                best_policy.append(Guess_actions[node_in])

                if Guess_actions[node_in] == 0:
                    probabilities = Q0[Guess_State[node_in], :].copy()
                    Guess_State[node_in] = np.random.choice(Game_State, p=probabilities)
                else:
                    probabilities = Q1[Guess_State[node_in], :].copy()
                    Guess_State[node_in] = np.random.choice(Game_State, p=probabilities)


            # print(best_policy)

            return best_policy


        def update_GameState(Old_GameState):
            new_GameState = Old_GameState.copy()
            Game_State_now = [0, 1, 2]
            for i in range(num_nodes):
                if Actions[i] == 0:
                    probabilities = Q0[new_GameState[i], :].copy()
                    new_GameState[i] = np.random.choice(Game_State, p=probabilities)
                else:
                    probabilities = Q1[new_GameState[i], :].copy()
                    new_GameState[i] = np.random.choice(Game_State, p=probabilities)
            return new_GameState


        all_game_states = []
        F_c = [np.mean(Actions)]

        x1 =[]
        x2=[]
        x3=[]
        rho_c =[]

        for sigma in [0.5]:
            delta = 0.1
            mu = 1 - sigma

            Q0 = np.array([
                [mu + delta, sigma - delta, 0],
                [mu + delta, 1 - (sigma + mu), sigma - delta],
                [0, mu + delta, sigma - delta]
            ])

            Q1 = np.array([
                [mu - delta, sigma + delta, 0],
                [mu - delta, 1 - (sigma + mu), sigma + delta],
                [0, mu - delta, sigma + delta]
            ])

            for x in range(num_iterations):


                node_best_policies = []
                for node in G.nodes:
                    node_best_policy = get_best_policy(node)
                    node_best_policies.append(node_best_policy)

                policy = node_best_policies.copy()




                for time in range(int(1 / (1-gamma))):
                    all_game_states.append(Current_GameState.copy())
                    Current_Actions = []
                    for node_policies_i in node_best_policies:
                        Current_Actions.append(node_policies_i[time])
                    Actions = Current_Actions.copy()
                    Current_GameState = update_GameState(Current_GameState.copy())
                    F_c.append(np.mean(Actions))

            proportions = []

            for moment in all_game_states:
                total = len(moment)
                count_0 = moment.count(0) / total
                count_1 = moment.count(1) / total
                count_2 = moment.count(2) / total
                proportions.append([count_0, count_1, count_2])

            proportions = np.array(proportions)


            average_value = []
            last_elements_PDG = proportions[-200:, 0]
            average_value.append(np.mean(last_elements_PDG))
            row1.append(np.mean(last_elements_PDG))
            last_elements_SDG = proportions[-200:, 1]
            average_value.append(np.mean(last_elements_SDG))
            row2.append(np.mean(last_elements_SDG))
            last_elements_SHG = proportions[-200:, 2]
            average_value.append(np.mean(last_elements_SHG))
            row3.append(np.mean(last_elements_SHG))


            x1.append(average_value[0])
            x2.append(average_value[1])
            x3.append(average_value[2])

    heatmap1.append(row1)
    heatmap2.append(row2)
    heatmap3.append(row3)

heatmap1 = gaussian_filter(heatmap1, sigma=2)
heatmap2 = gaussian_filter(heatmap2, sigma=2)
heatmap3 = gaussian_filter(heatmap3, sigma=2)

def save_heatmap_to_excel(heatmap, filename):

    df = pd.DataFrame(heatmap)

    df.to_excel(filename, index=False, header=False)


save_heatmap_to_excel(heatmap1, 'c_b_G1.xlsx')
save_heatmap_to_excel(heatmap2, 'c_b_G2.xlsx')
save_heatmap_to_excel(heatmap3, 'c_b_G3.xlsx')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24

x = np.linspace(0, 5, heatmap1.shape[1] )
y = np.linspace(0, 5, heatmap1.shape[0] )
plt.pcolormesh(x,y,heatmap1, cmap='jet', shading="gouraud")


plt.colorbar()


plt.xlabel(r"$b_1$")
plt.ylabel(r"$c$")
plt.xticks(ticks=[0,1,2,3,4,5])

plt.show()


x = np.linspace(0, 5, heatmap2.shape[1] )
y = np.linspace(0, 5, heatmap2.shape[0] )
plt.pcolormesh(x,y,heatmap2, cmap='jet', shading="gouraud")


plt.colorbar()


plt.xlabel(r"$b_1$")
plt.ylabel(r"$c$")
plt.xticks(ticks=[0,1,2,3,4,5])


plt.show()


x = np.linspace(0, 5, heatmap3.shape[1] )
y = np.linspace(0, 5, heatmap3.shape[0] )
plt.pcolormesh(x,y,heatmap3, cmap='jet', shading="gouraud")


plt.colorbar()


plt.xlabel(r"$b_1$")
plt.ylabel(r"$c$")
plt.xticks(ticks=[0,1,2,3,4,5])


plt.show()