from LQBP import LQBP
import numpy as np
import lxml.etree
#import random
import pandas as pd
from random import (choice, random, randint, randrange, uniform)

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
        #self.population_matrix = np.empty((0,6), int)
        self.best_chrm = []

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
        #self.population_matrix = np.empty((0,6), int)


        self.lqbp = LQBP(root)

        self.population = {} #dict that stores chromosomes as keys, and as values it has [x,y,z], where x,y are arrays of the param of the optimal solution, z is the optimal solution
        self.not_feasible = []
        self.best_chrm = [] #is an array of 4 elements storing the best chromosome, x, y, z

    def create_population(self): #function to be called to create generation 0 of chromosomes
        i = 0
        while i<self.population_size:
            tmp = np.array([],dtype=np.uint8)
            for j in range(self.lqbp.m+self.lqbp.ylength):
                tmp = np.append(tmp,[randint(0,1)])
            if self.get_feasible(tmp):
                i += 1
        self.gen_counter += 1

    def show_population(self): #function used to print the current population
        print("Generation",self.gen_counter)
        for k,v in self.population.items():
            tmp = list(k)
            print(tmp, end="->")
            print(v)

    def show_population_matrix(self):
        print("current population matrix is ")
        print(self.population_matrix)
        print("********************************************************")

    def get_feasible(self,tmp):
        x,y,z = self.lqbp.get_feasible(tmp)
        #if tuple(x) not in self.not_feasible:
        if isinstance(y,(list,pd.core.series.Series,np.ndarray)) and z <= 20: #if the operation has not been successfull, y is -1, so it doesn't enter this if condition
               if tuple(tmp) not in self.population.keys():
                   self.population[tuple(tmp)] = (x,y,z) #store chromosomes in dict as key, which has as value the solution found
                   return 1
               elif self.population[tuple(tmp)][2] < z:
                   self.population[tuple(tmp)] = (x,y,z)
        else:
               print("Not feasible",x)
               self.not_feasible.append(tuple(x)) #chromosome is not feasible
        return 0


    def find_best(self):
        max = float('-inf')
        for k,v in self.population.items():
              if max < v[2]:
                    max = v[2]
                    self.best_chrm = [k,v[0],v[1],v[2]]

    #1 updates the self.universal_chrm with new values
    def __main__(self):
        self.create_population()
        while self.gen_counter < self.max_generation:
         new_gen = np.array([])
         self.show_population()
         self.find_best()
         self.crossover(new_gen)
         self.mutation(new_gen)
         self.get_fitness(new_gen)
         self.selection()
         self.gen_counter += 1
         self.show_population()
        print("Best solution found:", self.best_chrm)

    def pop_random(self,lst):
        idx = randrange(0, len(lst))
        return list(lst.pop(idx))

    def crossover(self, new_gen): #parents contains chromosomes here
        print("Performing crossover!")

        parents = list(self.population.keys()) #obtaining key values (chrms) and putting them in parents as list

        chromosome_size = self.lqbp.m+self.lqbp.ylength

        tmp_parents = parents.copy() #used for obtaining couples

        pairs = [] #obtain random couples
        while len(tmp_parents) > 1:
              rand1 = self.pop_random(tmp_parents)
              rand2 = self.pop_random(tmp_parents)
              pair = [rand1, rand2]
              pairs.append(pair)
        print("Pair")
        print(pairs)

        for pair in pairs:
              print("new_gen", new_gen)
              if uniform(0,1) < self.crossover_prob:
                    crossover_point = randint(1,chromosome_size-1)
                    first_child = np.array(pair[0][:crossover_point])
                    second_child = np.array(pair[1][:crossover_point])
                    print("chrm size", chromosome_size)
                    print("crossover",crossover_point)
                    for i in range(chromosome_size-crossover_point):
                          print("i",i)
                          print("pair",pair)
                          print()
                          first_child = np.append(first_child,pair[1][chromosome_size-i-1])
                          second_child = np.append(second_child,pair[0][chromosome_size-i-1])
                    new_gen = np.append(new_gen,first_child)
                    new_gen = np.append(new_gen,second_child)
              else:
                    new_gen = np.append(new_gen,list(pair[0]))
                    new_gen = np.append(new_gen,list(pair[1]))
        print("new gen")
        print(new_gen)
        print("size",np.array([new_gen]).shape)
        new_gen = np.reshape(new_gen,(-1,chromosome_size))
        print("new gen")
        print(new_gen)

#step 5
    def mutation(self,new_gen):
        for i in range(len(new_gen)):
              for j in range(len(new_gen[i])):
                    if uniform(0,1) < self.mutation_prob:
                          new_gen[i][j] = 1-new_gen[i][j]

    def get_fitness(self,new_gen):
        for chrm in new_gen:
              self.get_feasible(chrm)

    #step 6
    def selection(self):
        self.rm_nonfeasible()
        self.population = self.sort_population()
        print("population conta")
        print(self.population)
        print("population size conta")
        print(self.population_size)
        print("real pop size")
        print(len(self.population))
        new_population = {}
        sum = 0

        for i in range(self.population_size):
              tmp = self.roulette_wheel_spin()
              new_population[tmp[0]] = (tmp[1],tmp[2],tmp[3])
        self.population = new_population

    def sort_population(self):
        return {k:v for k,v in sorted(self.population.items(), key=lambda item: item[1][2], reverse=True)}

    def rm_nonfeasible(self):
        pop = self.population.copy()
        for k,v in pop.items():
            x,y,z = self.lqbp.get_feasible(list(k),v[0])
            if not (isinstance(y,(list,pd.core.series.Series,np.ndarray)) and z <= 20):
               del self.population[k]

    def roulette_wheel_spin(self): #we have chromosomes as key and their fitness values as values
        print("POPULATION xandie")
        print(self.population)
        chrm_dict = self.population
        max_prob = 0
        #print(chrm_dict,'\n' ,type(chrm_dict))
        for i in chrm_dict:
            print(i, chrm_dict[i], type(chrm_dict))
            max_prob += abs(chrm_dict[i][2])
            print(chrm_dict[i][2])
        print("Sum of all the feasible scores", max_prob)
        new_chrm = ()
        pick = np.random.uniform(0, max_prob)
        current = 0
        print(max_prob)
        for chromosome in chrm_dict:
            current += abs(chrm_dict[chromosome][2])
            if current > pick:
                new_chrm = (chromosome,chrm_dict[chromosome][0],chrm_dict[chromosome][1],chrm_dict[chromosome][2])
                #self.population.pop(chromosome, None)
        print("picking up new ones from roulette:", new_chrm)
        self.population.pop(new_chrm[0],None)
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
