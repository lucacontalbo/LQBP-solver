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
