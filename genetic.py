from LQBP import LQBP
import lxml.etree

class Genetic:
	def __init__(self):
		self.population_size = int(input("Insert population size"))
		self.crossover_prob = int(input("Insert crossover probability"))
		self.mutation_prob = int(input("Insert mutation probability"))
		self.max_generation = int(input("Insert max number of generations"))
		self.gen_counter = 0
		lqbp = LQBP()

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
