from LQBP import LQBP
import numpy as np
import lxml.etree
import random

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
		self.population = np.array([], dtype = np.uint8)
		self.create_population()

	def create_population(self): #function to be called to create generation 0 of chromosomes
		i = 0
		while i<self.population_size:
			tmp = np.array([],dtype=np.uint8)
			for j in range(self.lqbp.m+self.lqbp.ylength):
				tmp = np.append(tmp,[random.randint(0,1)])
			if tmp not in self.population:
				self.population = np.append(self.population,tmp,axis=0)
				i += 1
		self.population = self.population.reshape(self.population_size,-1)
		self.lqbp.get_feasible(self.population[0])

	
	def calculating_pop_fitness(self,equation_inputs,pop): #calculating the fitness value for each solution in the current population
  		fitness=np.sum(pop*equation_inputs,axis=1)
  		return fitness

	def select_mating_pool(self,pop,fitness,num_parents):
  		parents = np.empty((num_parents, pop.shape[1]))
  		for parent_num in range(num_parents):
    		max_fitness_idx=np.where(fitness==numpy.max(fitness))
    		max_fitness_idx = max_fitness_idx[0][0]
    		parents[parent_num, :] = pop[max_fitness_idx, :]
    		fitness[max_fitness_idx] = -99999999999
  		return parents

	def crossover(self,parents, offspring_size):
    	offspring = numpy.empty(offspring_size) # The point at which crossover takes place between two parents. Usually it is at the center.
   		crossover_point = numpy.uint8(offspring_size[1]/2)
    	for k in range(offspring_size[0]):# Index of the first parent to mate.
        	parent1_idx = k%parents.shape[0]# Index of the second parent to mate.
        	parent2_idx = (k+1)%parents.shape[0] # The new offspring will have its first half of its genes taken from the first parent.
        	offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]# The new offspring will have its second half of its genes taken from the second parent.
        	offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    	return offspring

	def mutation(self,offspring_crossover):# Mutation changes a single gene in each offspring randomly.
    	for idx in range(offspring_crossover.shape[0]):# The random value to be added to the gene.
        	random_value = numpy.random.uniform(-1.0, 1.0, 1)
        	offspring_crossover[idx, 4] = offspring_crossover[idx, 4] + random_value
    	return offspring_crossover
