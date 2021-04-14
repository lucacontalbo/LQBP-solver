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
        self.population_matrix = np.empty((0,6), int)
        
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
        self.population_matrix = np.empty((0,6), int)


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
    def show_population_matrix(self):
        print("current population matrix is ")
        print(self.population_matrix)
        print("********************************************************")
            
            #_CODE UPDATE STARTS HERE____________________________________________________________________________________________________________________________________
    #return 1D array of feasibility scores
    def get_feasible(self,tmp, check = False):
        #print("current chromosome", type(tmp), '\n', tmp)
        #chrm = np.empty()
        if tuple(tmp) not in self.not_feasible:
            x,y,z = self.lqbp.get_feasible(tmp)
            if check == False:
                self.population_matrix = np.vstack((self.population_matrix, tmp))
            if isinstance(y,(list,pd.core.series.Series,np.ndarray)): #if the operation has not been successfull, y is -1, so it doesn't enter this if condition
                   self.population[tuple(tmp)] = (x,y,z) #store chromosomes in dict as key, which has as value the solution found
                   return 1
            else:
                self.not_feasible.append(tuple(tmp)) #chromosome is not feasible
        return 0
        
    #1 updates the self.universal_chrm with new values
    def __main__(self):
        while self.gen_counter < self.max_generation:
            self.show_population()
            if self.gen_counter == 0:
                self.create_population()
                self.show_population_matrix()
            else:
                self.crossover()
                self.show_population_matrix()
                self.mutation()
                self.show_population_matrix()
                self.selection()
                self.show_population_matrix()
            self.gen_counter += 1
        self.show_population()

        #it compares the chromosomes of current generation with the previous generation, returns the best chrm    
    
    #2 prform crossover and update self.universal_chrm
    def crossover(self): #parents contains chromosomes here
        print("Performing crossover!")
        parents = self.population_matrix
        #print("Parents", parents)
        
        chromosome_size = (self.population_matrix.shape[1])
        row_size = (self.population_matrix.shape[0]) #creating new generations parents -> offsprings // Here row size is handled by population
        offspring = np.copy(self.population_matrix)
        #print("Offprings are: ", '/n', offspring)
        
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
        
        #print('Offsprings from crossover: ','\n', offspring)
        self.population_matrix = offspring


#step 5
    def mutation(self):
        chrm = self.population_matrix
        print("Perfroming mutation! ")
        print('checking chrm')
        print(chrm)
        sampleList=np.random.choice(list(range(chrm.shape[0])),int(self.mutation_prob*chrm.shape[0]),replace=False)
        print(sampleList, type(sampleList))
        for i in sampleList:
            for j in range(len(chrm[0])): #edit
                chrm[i][j] = chrm[i][j]^1
        print("Chromosome after mutation: ", '\n', chrm)
        for i in chrm:
            self.get_feasible(i, True)
    
    #step 6
    def selection(self):
        print("Perform selection now!")
# =============================================================================
#         print("SELCTION NOW!")
#         chrm = np.array(list(self.population.keys()))
#         print("the size of populations is ", chrm.shape[1])
#         print(chrm, type(chrm))
#         
#         #chrm = self.population_matrix
#         dictionary = {}
#         feasiblity_scores = self.get_feasible(chrm) #doubt
#         print("feasibility score :", feasiblity_scores)
#         for i in range(chrm.shape[0]):
#             dictionary.update({i: feasiblity_scores[i]}) #calculate fitness value of all the chomosomes from sample
#         
#         print("dictionary with feasinbility scores", '\n',dictionary)
#         bestChromosomes_fitness_ascending = dict(sorted(dictionary.items(), key=lambda item: item[1]))  #sort from the dictionary top10 top8 top 12
# =============================================================================
        self.population_matrix = self.roulette_wheel_spin() #selecting the chromosomes on basis of wheel
        
        print("New Population matrix obtained!")
        

    def roulette_wheel_spin(self): #we have chromosomes as key and their fitness values as values
        chrm_dict = self.population
        max_prob = 0 
        #print(chrm_dict,'\n' ,type(chrm_dict))
        for i in chrm_dict:
            print(i, chrm_dict[i], type(chrm_dict))
            max_prob += abs(chrm_dict[i][2])
            print(chrm_dict[i][2])
        print("Sum of all the feasible scores", max_prob)
        new_chrm = np.empty((0,6), int)
        pick = np.random.uniform(0, max_prob)
        current = 0
        
        for chromosome in chrm_dict:
            current += abs(chrm_dict[chromosome][2])
            if current > pick:
                new_chrm = np.vstack((new_chrm, chromosome))
        print("picking up new ones from roulette:", new_chrm)
        return new_chrm
        #these are chromosomes for next generation
    
    #checks the feasibility of chrm and removes the not feasible ones from the current chromosomes
    def check_nonfeasible_chrm(self):
        chrm = self.population_matrix
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
        self.population_matrix=np.array(approved_chrm)
    
    def update_nonFeasible_list(self, waste_chrm): #the residual of the population is recieved here as np array, that will be merged with non_feasible
      if (self.not_feasible).size() == 0:
        self.not_feasible = np.append(self.notfeasible_chrm, waste_chrm, axis=0)
      else:
        self.not_feasible = np.vstack((self.notfeasible_chrm, waste_chrm))



#chrm = np.random.randint(2, size=(6,6))

#chrm = np.vstack((chrm,[1,2,1])


print('---------------------------------')
#x = Genetic()

g = Genetic('data.xml')
g.__main__()

"""
print(g.get_feasible())

print('---------------------------------')
chrm = g.crossover( 0.5)

print('---------------------------------')
chrm = g.mutation(chrm, 0.1)
print('---------------------------------')
print("Now selection", chrm)
chrm = g.selection(chrm)



print('rows', chrm.shape[0])
print('columns', chrm.shape[1])
print(chrm[2])
"""

