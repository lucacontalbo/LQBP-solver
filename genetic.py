from LQBP import LQBP
import numpy as np
import lxml.etree
import random
import pandas as pd

class Genetic:
	def __init__(self):
		self.population_size = int(input("Insert population size"))
		self.crossover_prob = int(input("Insert crossover probability"))
		self.mutation_prob = int(input("Insert mutation probability"))
		self.max_generation = int(input("Insert max number of generations"))
		self.gen_counter = 0
		self.lqbp = LQBP()
		self.population = np.array([]) #used to store the current population of chromosomes
		self.create_population()

	def __init__(self,path):
		with open(path) as f:
			lines = f.readlines()
		lines = '\n'.join(lines)
		root = lxml.etree.fromstring(lines)

		self.population_size = int(root.xpath('//app/popsize/cell/text()')[0])
		self.crossover_prob = float(root.xpath('//app/crossprob/cell/text()')[0])
		self.mutation_prob = float(root.xpath('//app/mutationprob/cell/text()')[0])
		self.max_generation = int(root.xpath('//app/maxgeneration/cell/text()')[0])
		self.gen_counter = 0


		self.lqbp = LQBP(root)

		self.population = {} #dict that stores chromosomes as keys, and as values it has [x,y,z], where x,y are arrays of the param of the optimal solution, z is the optimal solution
		self.not_feasible = []
		self.create_population()

	def create_population(self): #function to be called to create generation 0 of chromosomes
		i = 0
		while i<self.population_size:
			tmp = np.array([],dtype=np.uint8)
			for j in range(self.lqbp.m+self.lqbp.ylength):
				tmp = np.append(tmp,[random.randint(0,1)])
			if tuple(tmp) not in self.not_feasible:
				x,y,z = self.lqbp.get_feasible(tmp)
				if isinstance(y,(list,pd.core.series.Series,np.ndarray)): #if the operation has not been successfull, y is -1, so it doesn't enter this if condition
					self.population[tuple(tmp)] = (x,y,z) #store chromosomes in dict as key, which has as value the solution found
					i += 1
				else:
					self.not_feasible.append(tuple(tmp)) #chromosome is not feasible
		self.debug_pop()

	def debug_pop(self): #function used only for debugging purposes
		for k,v in self.population.items():
			tmp = list(k)
			#tmp = tmp.reshape(-1,self.lqbp.m+self.lqbp.ylength)
			print(tmp, end="->")
			print(v)

# TODO best_chromosome_for_mating,crossover,mutation
	# fitness value is calculated by the simplex method to obtain the feasibility and fitness value of the chromosome
	# Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
	def best_chromosome_for_mating(self, new_population, fitness, num_parents_mating):
		parents = np.empty((num_parents_mating, new_population.shape[1]))
		for parent_num in range(num_parents_mating):
        		max_fitness_idx = np.where(fitness == np.max(fitness))
        		max_fitness_idx = max_fitness_idx[0][0]
        		parents[num_parents_mating, :] = new_population[max_fitness_idx, :]
        		fitness[max_fitness_idx] = -99999999999
		return parents

	def crossover(self, parents, offspring_size, crossover_prob):
			offspring = np.empty(offspring_size)
			# crossover_point_prob - The point at which crossover takes place between two parents.
			crossover_point = np.uint8(offspring_size[1]/2)
			for k in range(offspring_size[0]):
					parent1_idx = k % parents.shape[0]  # Index of the first parent to mate.
					parent2_idx = (k+1) % parents.shape[0] # Index of the second parent to mate.
					offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point] # The new offspring will have its first half of its genes taken from the first parent.
					offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:] # The new offspring will have its second half of its genes taken from the second parent.
			return offspring

	def mutation(self,offspring_crossover,mutation_prob):# Mutation changes a single gene in each offspring randomly.
		for idx in range(offspring_crossover.shape[0]):
			offspring_crossover[idx, 4] = offspring_crossover[idx, 4] + mutation_prob #mutation prob is already defined
		return offspring_crossover

	def selection():
		for generation in range(num_generations):
			fitness=cal_pop_fitness(self,inputs_equation,new_population)
			parents=select_mating_pool(self,new_population,fitness,self.population_size) # selecting the best parent in the population for matching
			offspring_crossover=crossover(self,parents,offspring_size=(pop_size[0]-parents.shape[0],num_weights)) # generating the next generation
			offspring_mutation=mutation(self,offspring_crossover) # Adding some variation to the offspring using mutation
			new_population[0:parents.shape[0],:]=parents # creating the new popultion based on the parents and offspring
			new_population[parents.shape[0]:,:]=offspring_mutation
			print("Best result:",np.max(np.sum(new_population*inputs_equation,axis=1))) # The best result in the current iteration

	def Termination():
			# Getting the best solution after iterating all the generations
			fitness=cal_pop_fitness(self,inputs_equation,new_population)
			best_match=np.where(fitness==np.max(fitness))
			print('Best Solution',new_population[best_match,:])
			print('Best solution fitness',fitness[best_match])

