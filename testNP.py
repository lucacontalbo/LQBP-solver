# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 00:42:30 2021

@author: sande
"""

# importing library
import numpy as np
l = np.array([[1,0,0,1,0,1],[1,0,1,1,0,0],[0,0,1,1,0,1],[0,1,1,1,1,0]])
x = np.empty(6)
for i in l:
  x = np.vstack(x,np.array(i)) 
print(x)
