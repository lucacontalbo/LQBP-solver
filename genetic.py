class Genetic:
	def __init__(self):
		self.population_size = int(input("Insert population size"))
		self.crossover_prob = int(input("Insert crossover probability"))
		self.mutation_prob = int(input("Insert mutation probability"))
		self.max_generation = int(input("Insert max number of generations"))
		self.gen_counter = 0

	def __init__(self,popsize,crossprob,mutationprob,maxgeneration):
		self.population_size = popsize
		self.crossover_prob = crossprob
		self.mutation_prob = mutationprob
		self.max_generation = maxgeneration
		self.gen_counter = 0
