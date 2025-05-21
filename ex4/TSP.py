import numpy as np
import matplotlib.pyplot as plt
import random
import math


class TSP:
    def __init__(self, num_cities=20, random_seed=None):
        """初始化TSP问题，生成随机城市坐标"""
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        self.num_cities = num_cities
        self.cities = np.random.rand(num_cities, 2)  # 随机生成城市坐标
        self.distance_matrix = self._calculate_distance_matrix()

    def _calculate_distance_matrix(self):
        """计算城市间距离矩阵"""
        matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                dist = np.sqrt(np.sum((self.cities[i] - self.cities[j]) ** 2))
                matrix[i, j] = dist
                matrix[j, i] = dist
        return matrix

    def evaluate(self, route):
        """评估路径的总长度"""
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += self.distance_matrix[route[i], route[i + 1]]
        # 回到起点
        total_distance += self.distance_matrix[route[-1], route[0]]
        return total_distance

    def plot_route(self, route, title="TSP Route"):
        """可视化TSP路径"""
        plt.figure(figsize=(10, 8))

        # 绘制城市点
        plt.scatter(self.cities[:, 0], self.cities[:, 1], c='blue', s=50)

        # 绘制路径
        route_coords = self.cities[route]
        plt.plot(np.append(route_coords[:, 0], route_coords[0, 0]),
                 np.append(route_coords[:, 1], route_coords[0, 1]),
                 'r-')

        # 添加城市编号
        for i, (x, y) in enumerate(self.cities):
            plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')

        plt.title(title)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class EvolutionaryAlgorithm:
    def __init__(self, tsp, population_size=100, generations=500,
                 crossover_rate=0.8, mutation_rate=0.2, elitism_rate=0.1):
        """初始化进化算法参数"""
        self.tsp = tsp
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.elitism_count = int(population_size * elitism_rate)

    def initialize_population(self):
        """初始化种群"""
        population = []
        for _ in range(self.population_size):
            # 随机生成一个有效路径（城市排列）
            individual = list(range(self.tsp.num_cities))
            random.shuffle(individual)
            population.append(individual)
        return population

    def evaluate_population(self, population):
        """评估种群中每个个体的适应度（路径长度的倒数）"""
        fitness_scores = []
        for individual in population:
            distance = self.tsp.evaluate(individual)
            # 路径越短，适应度越高
            fitness = 1.0 / distance
            fitness_scores.append(fitness)
        return fitness_scores

    def select_parents(self, population, fitness_scores):
        """使用轮盘赌选择方法选择父代"""
        total_fitness = sum(fitness_scores)
        selection_probs = [fitness / total_fitness for fitness in fitness_scores]

        parents = []
        for _ in range(self.population_size):
            # 选择两个父代
            parent1 = self._roulette_wheel_selection(population, selection_probs)
            parent2 = self._roulette_wheel_selection(population, selection_probs)
            parents.append((parent1, parent2))
        return parents

    def _roulette_wheel_selection(self, population, probabilities):
        """轮盘赌选择方法"""
        r = random.random()
        cumulative_prob = 0
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if cumulative_prob >= r:
                return population[i]
        return population[-1]

    def ordered_crossover(self, parent1, parent2):
        """有序交叉操作，生成两个子代"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        # 随机选择两个交叉点
        start, end = sorted(random.sample(range(len(parent1)), 2))

        # 初始化子代为空
        child1 = [-1] * len(parent1)
        child2 = [-1] * len(parent2)

        # 复制交叉点之间的部分
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]

        # 剩余城市的顺序
        remaining1 = [city for city in parent2 if city not in child1[start:end]]
        remaining2 = [city for city in parent1 if city not in child2[start:end]]

        # 填充剩余城市
        index = 0
        for i in range(len(child1)):
            if child1[i] == -1:
                child1[i] = remaining1[index]
                index += 1

        index = 0
        for i in range(len(child2)):
            if child2[i] == -1:
                child2[i] = remaining2[index]
                index += 1

        return child1, child2

    def swap_mutation(self, individual):
        """交换变异操作"""
        if random.random() > self.mutation_rate:
            return individual.copy()

        # 随机选择两个位置进行交换
        pos1, pos2 = random.sample(range(len(individual)), 2)
        individual = individual.copy()
        individual[pos1], individual[pos2] = individual[pos2], individual[pos1]
        return individual

    def evolve(self):
        """执行进化算法"""
        # 初始化种群
        population = self.initialize_population()

        # 记录每代的最佳适应度和平均适应度
        best_fitness_history = []
        avg_fitness_history = []
        best_individual = None
        best_fitness = -float('inf')

        for generation in range(self.generations):
            # 评估种群
            fitness_scores = self.evaluate_population(population)

            # 记录最佳个体和适应度
            current_best_idx = fitness_scores.index(max(fitness_scores))
            current_best_fitness = fitness_scores[current_best_idx]
            current_best_individual = population[current_best_idx]

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best_individual.copy()

            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(sum(fitness_scores) / len(fitness_scores))

            # 精英保留
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[
                            :self.elitism_count]
            elites = [population[i] for i in elite_indices]

            # 选择父代
            parents = self.select_parents(population, fitness_scores)

            # 生成子代
            offspring = []
            for parent1, parent2 in parents:
                child1, child2 = self.ordered_crossover(parent1, parent2)
                child1 = self.swap_mutation(child1)
                child2 = self.swap_mutation(child2)
                offspring.extend([child1, child2])

            # 新种群由精英和子代组成
            population = elites + offspring[:self.population_size - self.elitism_count]

            # 打印进度
            if generation % 50 == 0:
                best_distance = 1.0 / best_fitness
                print(f"Generation {generation}: Best Distance = {best_distance:.4f}")

        return best_individual, best_fitness_history, avg_fitness_history


# 主函数
def main():
    # 创建TSP问题实例
    tsp = TSP(num_cities=20, random_seed=42)

    # 创建进化算法实例
    ea = EvolutionaryAlgorithm(
        tsp=tsp,
        population_size=100,
        generations=500,
        crossover_rate=0.8,
        mutation_rate=0.2,
        elitism_rate=0.1
    )

    # 运行进化算法
    best_route, best_fitness_history, avg_fitness_history = ea.evolve()

    # 计算最佳路径长度
    best_distance = tsp.evaluate(best_route)
    print(f"\n最佳路径长度: {best_distance:.4f}")
    print(f"最佳路径: {best_route}")

    tsp.plot_route(best_route, f"Optimal TSP Route (Distance: {best_distance:.4f})")

    # 可视化适应度变化
    plt.figure(figsize=(10, 6))
    plt.plot([1.0 / f for f in best_fitness_history], label='Best Route Distance')
    plt.plot([1.0 / f for f in avg_fitness_history], label='Average Route Distance')
    plt.xlabel('Generation')
    plt.ylabel('Route Distance')
    plt.title('Change in Route Distance During Evolution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()