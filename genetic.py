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
		lqbp = LQBP()
		self.population = np.array([]) #used to store the current population of chromosomes

	def __init__(self,path):
		with open(path) as f:
			lines = f.readlines()
		lines = '\n'.join(lines)
		root = lxml.etree.fromstring(lines)

		self.population_size = root.xpath('//app/popsize/cell/text()')
		self.crossover_prob = root.xpath('//app/crossprob/cell/text()')
		self.mutation_prob = root.xpath('//app/mutationprob/cell/text()')
		self.max_generation = root.xpath('//app/maxgeneration/cell/text()')
		self.gen_counter = 0

		lqbp = LQBP(root)
		self.population = []

	def create_population(self): #function to be called to create generation 0 of chromosomes
		i = 0
		while i<self.population_size:
			tmp = np.array([])
			for j in range(self.lqbp.m+self.lqbp.ylength):
				tmp = np.append(tmp,[random.randint(0,1)])
			if tmp not in self.population:
				self.population = np.append(self.population,tmp)
				i += 1

