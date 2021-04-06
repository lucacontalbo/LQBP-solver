from LQBP import LQBP
import numpy as np
import lxml.etree
import random
import pandas as pd
from random import (choice, random, randint)

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

	def crossover(self,chrm,parents,self.crossover_prob):
			chromosome_size = (np.shape(parents)[1])
			row_size = np.shape(parents)[0] #creating new generations parents -> offsprings
			offspring = np.copy(parents)
			selected_parents_indexes=np.random.choice(list(range(row_size)),int(crossover_prob*row_size),replace=False)
			crossover_point = int((np.shape(parents)[1])/2)
			for i in range(len(selected_parents_indexes)):
				if i!=len(selected_parents_indexes)-1:
					offspring[selected_parents_indexes[i]] = list(parents[selected_parents_indexes[i]]
					[:crossover_point]) + list(parents[selected_parents_indexes[i+1]]
					[:crossover_point-1:-1])
			else:
					offspring[selected_parents_indexes[i]] = list(parents[selected_parents_indexes[i]]
					[:crossover_point]) + list(parents[selected_parents_indexes[0]]
					[:crossover_point-1:-1])
			return offspring


	def mutation(self,chrm,mutation_prob):
    		sampleList=np.random.choice(list(range(len(chrm)),int(mutation_prob*len(chrm)),replace=False))
			for i in self.lqbp.chrm:
    				self.lqbp.chrm[i]=self.lqbp.chrm[i]^1
			return self.lqbp.chrm

	def selection(self,chrm,self.population,fitness):
			dictionary = {}
			for i in chromosome:
					dictionary.append(chromosome, fitness(chromosome))#calculate fitness value of all the chomosomes from sample
			bestChromosomes_fitness_ascending=dict(sorted(dictionary.items(), key=lambda item: item[1]))  #sort from the dictionary top10 top8 top 12
			selectedChromosomes = roulette_wheel_spin(bestChromosomes_fitness_ascending, self.population_size) #selected best chromosome based on wheel
			def selectOne(self, bestChromosomes_fitness_ascending,population):
				max = sum([c.bestChromosomes_fitness_ascending for c in population])
				pick = random.uniform(0, max)
				current = 0
				for chromosome in population:
					current += chromosome.bestChromosomes_fitness_ascending
				if current > pick:
				return chromosome
			#these are chromosomes for next generation
			return selectedChromosomes



