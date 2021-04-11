from LQBP import LQBP
import numpy as np
import lxml.etree
#import random
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
        self.not_feasible = []
        self.population = {}

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

    def create_population(self): #function to be called to create generation 0 of chromosomes
        i = 0
        while i<self.population_size:
            tmp = np.array([],dtype=np.uint8)
            for j in range(self.lqbp.m+self.lqbp.ylength):
                tmp = np.append(tmp,[randint(0,1)])
            if self.get_feasible(tmp):
                i += 1
        self.show_population()

    def show_population(self): #function used to print the current population
        print("Generation",self.gen_counter)
        for k,v in self.population.items():
            tmp = list(k)
            #tmp = tmp.reshape(-1,self.lqbp.m+self.lqbp.ylength)
            print(tmp, end="->")
            print(v)
            
            #_CODE UPDATE STARTS HERE____________________________________________________________________________________________________________________________________
    #return 1D array of feasibility scores
    def get_feasible(self,tmp):
        if tuple(tmp) not in self.not_feasible:
        	x,y,z = self.lqbp.get_feasible(tmp)
       		if isinstance(y,(list,pd.core.series.Series,np.ndarray)): #if the operation has not been successfull, y is -1, so it doesn't enter this if condition
       			self.population[tuple(tmp)] = (x,y,z) #store chromosomes in dict as key, which has as value the solution found
       			return 1
        	else:
        		self.not_feasible.append(tuple(tmp)) #chromosome is not feasible
        return 0
        
        """print("fetching get_feasibility scores:")
        values = []
        optimum_value = 0
        print("equation: 2x + 4y - 4z + 3w -8a +7b - 2 = 0") # 1 0 1 1 1 1
        for i in range(chrm.shape[0]):
            print("interation", i, chrm[i]) 
            x = (2*(chrm[i][0]) + 2*(chrm[i][1]) - 4*(chrm[i][2]) + 3*(chrm[i][3]) -8*(chrm[i][4]) + 2*(chrm[i][5] + 2))
            values.append(abs(optimum_value -x)) #how far is our score from the optimum score
        print("FSCORES ARE: ", values)
        return values"""
        
    #1 updates the self.universal_chrm with new values
    def __main__(self):
        while self.gen_counter < self.max_generation:
            if self.gen_counter == 0:
                self.create_population()
            else:
                self.crossover()
                self.mutation()
                self.selection()
            self.gen_counter += 1

        #it compares the chromosomes of current generation with the previous generation, returns the best chrm    
        
    '''if len(self.population_size) == 0:
                self.create_population()
        else:
                if get_feasible(self.universal_chrm) > get_feasible(chrm): #need the fitness score calculation for chromosome DOUBT about function
                        return self.universal_chrm
                else:
                        self.universal_chrm = chrm
        return chrm'''
    
    #2 prform crossover and update self.universal_chrm
    def crossover(self): #parents contains chromosomes here
        print("Performing crossover!")
        chromosome_size = (self.population.shape[1])
        row_size = self.population.shape[0] #creating new generations parents -> offsprings // Here row size is handled by population
        offspring = np.copy(self.population)
        print("Offprings are: ", '/n', offspring)
        
        selected_parents_indexes=np.random.choice(list(range(row_size)),int(self.crossover_prob*row_size),replace=False)
        print("indexes of selected parents", selected_parents_indexes)
        
        crossover_point = int(chromosome_size/2)
        
        for i in range(len(selected_parents_indexes)):
            if i!=len(selected_parents_indexes)-1:
                offspring[selected_parents_indexes[i]] = list(parents[selected_parents_indexes[i]]
                [:crossover_point]) + list(parents[selected_parents_indexes[i+1]]
                [:crossover_point-1:-1])
            else:
                offspring[selected_parents_indexes[i]] = list(parents[selected_parents_indexes[i]]
                [:crossover_point]) + list(parents[selected_parents_indexes[0]]
                [:crossover_point-1:-1])
        
        print('Offsprings from crossover: ','\n', offspring)
        self.population = offspring


#step 5
    def mutation(self):
        chrm = self.population
        print("Perfroming mutation! ")
        sampleList=np.random.choice(list(range(chrm.shape[0])),int(mutation_prob*chrm.shape[0]),replace=False)
        print(sampleList, type(sampleList))
        for i in range(chrm.shape[0]):
            for j in range(len(sampleList)): #edit
                chrm[i][j] = chrm[i][j]^1
        print("Chromosome after mutation: ", '\n', chrm)
        self.population= self.check_nonfeasible_chrm(chrm) #doubt
        return chrm
    
    #step 6
    def selection(self):
        print("Perfroming selection!")
        chrm = self.population
        dictionary = {}
        feasiblity_scores = self.get_feasible(chrm) #doubt
        print("feasibility score :", feasiblity_scores)
        for i in range(chrm.shape[0]):
            dictionary.update({i: feasiblity_scores[i]}) #calculate fitness value of all the chomosomes from sample
        
        print("dictionary with feasinbility scores", '\n',dictionary)
        bestChromosomes_fitness_ascending=dict(sorted(dictionary.items(), key=lambda item: item[1]))  #sort from the dictionary top10 top8 top 12
        selectedChromosomes_index = self.roulette_wheel_spin(bestChromosomes_fitness_ascending) #selecting the chromosomes on basis of wheel
        
        #resultant chromosomes stored in res
        res= np.empty((0,chrm.shape[1]), int)
        for i in range(len(selectedChromosomes_index)):
            res = np.vstack((res, chrm[selectedChromosomes_index[i]]))
        print("Resulting chromosomes after wheel spin", res)
        self.population = res
        

    def roulette_wheel_spin(self,chrm_dict): #we have chromosomes as key and their fitness values as values
        max_prob = sum(chrm_dict.values()) #this returns an array of all the 
        new_chrm = []
        pick = random.uniform(0, max_prob)
        current = 0
        for chromosome in chrm_dict:
            current += chrm_dict[chromosome]
            if current > pick:
                new_chrm.append(chromosome)
        print("picking up new ones from roulette:", new_chrm)
        return new_chrm
        #these are chromosomes for next generation
    
    #checks the feasibility of chrm and removes the not feasible ones from the current chromosomes
    def check_nonfeasible_chrm(self):
        chrm = self.population
        approved_chrm = []
        flag= False
          #np.empty((0,6),int)
        for i in chrm:
            flag = False 
            for j in self.not_feasible:
              if list(i) == list(j):
                flag = True #copy is found, exit and dont append
            if flag == False:
              approved_chrm.append(i)
        approved_chrm = np.reshape(approved_chrm, (len(approved_chrm), chrm.shape[0]))
        print("approved_chrm", approved_chrm)
        self.population=np.array(approved_chrm)
    
    def update_nonFeasible_list(waste_chrm): #the residual of the population is recieved here as np array, that will be merged with non_feasible
      if (self.not_feasible).size() == 0:
        self.not_feasible = np.append(self.notfeasible_chrm, waste_chrm, axis=0)
      else:
        self.not_feasible = np.vstack((self.notfeasible_chrm, waste_chrm))


"""

chrm = np.random.randint(2, size=(6,6))

#chrm = np.vstack((chrm,[1,2,1])


print('---------------------------------')
g = Genetic()
print(g.get_feasible(chrm))

print('---------------------------------')
chrm = g.crossover(chrm, 0.5)

print('---------------------------------')
chrm = g.mutation(chrm, 0.1)
print('---------------------------------')
print("Now selection", chrm)
chrm = g.selection(chrm)



print('rows', chrm.shape[0])
print('columns', chrm.shape[1])
print(chrm[2])

"""
