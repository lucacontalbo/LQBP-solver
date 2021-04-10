#!/usr/bin/env python
# coding: utf-8

# In[133]:


import numpy as np
import lxml.etree
import random
import pandas as pd


class Genetic:
    def __init__(self):
        self.universal_chrm = np.empty((0,6), int)
        self.notfeasible_chrm = np.empty((0,6), int)

    #return 1D array of feasibility scores
    def get_feasible(self):
         x,y,z = self.lqbp.get_feasible(tmp)
                if isinstance(y,(list,pd.core.series.Series,np.ndarray)): #if the operation has not been successfull, y is -1, so it doesn't enter this if condition
                    self.population[tuple(tmp)] = (x,y,z) #store chromosomes in dict as key, which has as value the solution found
                    i += 1
                else:
                    self.not_feasible.append(tuple(tmp)) #chromosome is not feasible
        print(x,y,z)
        
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
        for i in self.max_generation:
            if i == 0:
                self.create_population()
            else:
                self.crossover()
                self.mutation()
                self.selection()
                
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


"""
print('rows', chrm.shape[0])
print('columns', chrm.shape[1])
print(chrm[2])

"""








# In[127]:


chrm = np.random.randint(2, size=(6,6))
print("Chromosomes are :", chrm)
index_array = [1]
print("given index are 1 & 3rd, what is resulting np array!")
res=np.empty
res = chrm[index_array[0]]
print("res", res)
for i in range(1,len(index_array)):
    res = np.vstack((res, chrm[index_array[i]]))

print(res)

#x = np.vstack((chrm[0],chrm[1]))
#x = np.vstack((x, chrm[3]))
#print(x)


# In[130]:


print(range(0,1))

