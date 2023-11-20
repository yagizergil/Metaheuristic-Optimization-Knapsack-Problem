import numpy as np
import matplotlib.pyplot as plt
import random




class SlimeMoldOptimization:
    def __init__(self, knapsack_capacity, weights, values, num_particles, num_iterations):
        self.knapsack_capacity = knapsack_capacity
        self.weights = weights
        self.values = values
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.best_fitnesses = []

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


        print("Knapsack Capacity:", self.knapsack_capacity)
        print("Total Weight:", best_total_weight)
        print("Best items:", best_items)
        print("Best fitness:", best_total_value)

        return best_items, best_total_value , best_fitnesses

    def _evaluate_fitness(self, particle):
        total_value = np.sum(particle * self.values)
        total_weight = np.sum(particle * self.weights)

        if total_weight > self.knapsack_capacity:
            total_value = float('-inf')

        return total_value

class HarrisHawkMetaheuristicAlgorithm:
    def __init__(self, population_size, solution_length, num_iterations, roosting_factor, exploration_factor,
                 weights, prices, capacity):
        self.population_size = population_size
        self.solution_length = solution_length
        self.num_iterations = num_iterations
        self.roosting_factor = roosting_factor
        self.exploration_factor = exploration_factor
        self.weights = weights
        self.prices = prices
        self.capacity = capacity
        self.population = []

    def generate_initial_population(self):
        for _ in range(self.population_size):
            solution = [random.randint(0, 1) for _ in range(self.solution_length)]
            self.population.append(solution)

    def calculate_fitness(self, solution):
        total_weight = np.sum(solution * self.weights)
        total_price = np.sum(solution * self.prices)

        if total_weight > self.capacity:
            return 0
        else:
            return total_price

    def find_best_solution(self):
        best_solution = self.population[0]
        best_fitness = self.calculate_fitness(best_solution)

        for solution in self.population:
            fitness = self.calculate_fitness(solution)
            if fitness > best_fitness:
                best_solution = solution
                best_fitness = fitness

        return best_solution, best_fitness

    def roosting(self, solution):
        new_solution = []
        for gene in solution:
            if random.random() < self.roosting_factor:
                new_gene = 1 if gene == 0 else 0
                new_solution.append(new_gene)
            else:
                new_solution.append(gene)
        return new_solution

    def exploration(self, solution):
        new_solution = []
        for gene in solution:
            if random.random() < self.exploration_factor:
                new_gene = random.randint(0, 1)
                new_solution.append(new_gene)
            else:
                new_solution.append(gene)
        return new_solution

    def optimize(self):
        self.generate_initial_population()
        iteration_results_hawk = []
        best_solution, best_fitness = self.find_best_solution()

        for _ in range(self.num_iterations):
            new_population = []
            for solution in self.population:
                if solution == best_solution:
                    new_solution = self.roosting(solution)
                else:
                    new_solution = self.exploration(solution)

                new_population.append(new_solution)

            self.population = new_population

            current_best_solution, current_best_fitness = self.find_best_solution()

            if current_best_fitness > best_fitness:
                best_solution = current_best_solution
                best_fitness = current_best_fitness
            iteration_results_hawk.append(current_best_fitness)  # Her iterasyonda en iyi uygunluk değerini ekle

        # En iyi çözümü ve uygunluk değerini, ayrıca iterasyon sonuçlarını döndür
        return best_solution, best_fitness, iteration_results_hawk

# Örnek kullanım
weights = np.array([2,1,3,2,5,7,9,1,4,6,13,12,20,8,10,4,6,15,3,11,17,9,5,14,18,7,16,12,8,6,2,1,3,2,5,7,9,1,4,6,13,12,20,8,10,4,6,15])
values = np.array([12,10,20,15,28,19,56,43,12,18,21,14,18,32,24,8,22,30,16,27,25,13,11,35,40,17,29,23,9,26,12,10,20,15,28,19,56,43,12,18,21,14,18,32,24,8,22,30])
knapsack_capacity = 100
num_particles = 100
num_iterations = 400
population_size = 50
roosting_factor = 0.000001
exploration_factor = 50

smo = SlimeMoldOptimization(knapsack_capacity, weights, values, num_particles, num_iterations)
best_items, best_fitness , best_fitnesses= smo.optimize()

harris_hawk_algorithm = HarrisHawkMetaheuristicAlgorithm(population_size = population_size, solution_length=len(weights), num_iterations=num_iterations, roosting_factor=roosting_factor, exploration_factor=exploration_factor, weights=weights, prices=values, capacity=knapsack_capacity)
best_solution, best_fitness, iteration_results = harris_hawk_algorithm.optimize()


print("Harris Hawk Metaheuristic Algorithm:")
print("Solution:", best_solution)
print("Best Fitness:", best_fitness)

plt.figure(figsize=(12, 9))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_iterations + 1), best_fitnesses)
plt.xlabel('Iterasyon')
plt.ylabel('En İyi Uygunluk Değeri')
plt.title('Slime Mould Algorithm - En İyi Uygunluk Değerinin İterasyona Göre Değişimi')


plt.show(block=False)

plt.subplot(1, 2, 2)
plt.plot(range(1, num_iterations + 1), iteration_results, color='blue')
plt.xlabel('Iterasyon')
plt.ylabel('En İyi Uygunluk Değeri')
plt.title('Harris Hawk Algorithm - En İyi Uygunluk Değerinin İterasyona Göre Değişimi')
plt.show()


plt.plot(range(1, smo.num_iterations + 1), best_fitnesses, color='blue', label='Slime Mold Optimization')
plt.plot(range(1, num_iterations + 1), iteration_results, color='red', label='Harris Hawk Algorithm')

plt.xlabel('Iterasyon')
plt.ylabel('En İyi Uygunluk Değeri')
plt.title('Slime Mold vs Harris Hawk')
plt.legend()

plt.tight_layout()
plt.show()