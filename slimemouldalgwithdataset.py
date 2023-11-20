import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
class SlimeMoldOptimization:
    def __init__(self, knapsack_capacity, weights, values, num_particles, num_iterations):
        self.knapsack_capacity = knapsack_capacity
        self.weights = weights
        self.values = values
        self.num_particles = num_particles
        self.num_iterations = num_iterations

    def optimize(self):
        num_items = len(self.weights)
        particles = np.random.randint(2, size=(self.num_particles, num_items))
        particles = particles.astype(float)
        best_position = particles[0].copy()
        best_fitness = self._evaluate_fitness(best_position)
        best_fitnesses = []
        best_total_value = 0
        best_total_weight = 0

        for _ in range(self.num_iterations):
            for i in range(self.num_particles):
                fitness = self._evaluate_fitness(particles[i])

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_position = particles[i].copy()



            best_items = [item for item, select in zip(range(num_items), best_position) if select == 1]
            total_weight = sum(self.weights[item] for item in best_items)
            if total_weight <= self.knapsack_capacity and best_fitness > best_total_value:
                best_total_value = best_fitness
                best_total_weight = total_weight

            best_fitnesses.append(best_fitness)

            for i in range(self.num_particles):
                for j in range(num_items):
                    if np.random.random() < 0.5:
                        particles[i, j] = 1.0
                    else:
                        particles[i, j] = 0.0

        best_items = [item for item, select in zip(range(num_items), best_position) if select == 1]
        total_weight = sum(self.weights[item] for item in best_items)
        print(best_fitnesses)

        plt.plot(range(1, self.num_iterations + 1), best_fitnesses)
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        plt.title('Iteration vs. Best Fitness')
        plt.show()

        print("Knapsack Capacity:", self.knapsack_capacity)
        print("Total Weight:", best_total_weight)
        print("Best items:", best_items)
        print("Best fitness (Total value):", best_total_value)

        return best_items, best_total_value

    def _evaluate_fitness(self, particle):
        total_value = np.sum(particle * self.values)
        total_weight = np.sum(particle * self.weights)

        if total_weight > self.knapsack_capacity:
            total_value = float('-inf')

        return total_value

data = pd.read_csv('knapsack.csv')
num_iterations = 100
n_iterations = 100
# Dataset içindeki satır sayısını elde etme
num_rows = data.shape[0]

# Rastgele bir satır seçme
random_index = random.randint(0, num_rows - 1)
random_row = data.iloc[random_index]

# Seçilen satırın verilerini al
weights = np.array(random_row['Weights'].strip('[]').split(), dtype=int)
values = np.array(random_row['Prices'].strip('[]').split(), dtype=int)
knapsack_capacity = random_row['Capacity']

# Sonucu yazdırma
print("Satır", random_index + 1)
print('Weights:', weights)
print('Prices:', values)
print('Capacity:', knapsack_capacity)


print("Seçilen Satır:")
print(random_row)
print()
num_particles = 20
num_iterations = 400

smo = SlimeMoldOptimization(knapsack_capacity, weights, values, num_particles, num_iterations)
best_items, best_fitness = smo.optimize()
