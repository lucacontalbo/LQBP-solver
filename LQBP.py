import numpy as np
import lxml.etree
import random
import pandas as pd

class LQBP:
	def __init__(self):
		self.xlength = int(input("Number of x variables"))
		self.ylength = int(input("Number of y variables"))
		print("Insert"+self.xlength+"values into c")
		self.c = np.array([])
		self.d = np.array([])
		for i in range(self.xlength):
			self.c = np.append(self.c,int(input()))
		print("Insert"+self.ylength+"values into d")
		self.d = np.array([])
		tmp = self.xlength+self.ylength
		for i in range(self.ylength):
			self.d = np.append(self.d,int(input()))
		print("Insert values into a"+tmp+"x"+tmp+"Q matrix")
		self.Q = np.zeros((tmp,tmp))
		for i in range(tmp):
			for j in range(tmp):
				self.Q[i][j] = input("Insert value into cell ("+i+","+j+")")
		print("About constraints...")
		self.m = input("Insert r length")
		print("Insert"+self.m+"values into r")
		self.r = np.array([])
		for i in range(self.m):
			r = np.append(r,int(input()))
		self.A = np.zeros((self.m,self.xlength))
		self.B = np.zeros((self.m,self.ylength))
		print("Insert values into a"+self.m+"x"+self.xlength+"X matrix")
		for i in range(self.m):
			for j in range(self.xlength):
				A[i][j] = int(input("Insert value into cell ("+i+","+j+")"))
		print("Insert values into a"+self.m+"x"+self.ylength+"B matrix")
		for i in range(self.m):
			for j in range(self.ylength):
				B[i][j] = int(input("Insert value into cell ("+i+","+j+")"))
		print("Insert 1 if you want to maximize leader function, 0 to minimize")
		self.max1 = int(input())
		while(self.max1 != 0 and self.max1 != 1):
			print("Wrong number inserted. Please repeat")
			self.max1 = int(input())
		print("Insert 1 if you want to maximize follower function, 0 to minimize")
		self.max2 = int(input())
		while(self.max2 != 0 and self.max2 != 1):
			print("Wrong number inserted. Please repeat")
			self.max2 = int(input())
		self.ubound = 50

	def __init__(self,root):
		#this part works only with a properly defined xml file passed as input. It is the constructor to use for testing

		self.xlength = root.xpath('//app/xlen/cell/text()')
		self.ylength = root.xpath('//app/ylen/cell/text()')
		self.a = root.xpath('//app/a/cell/text()')
		self.b = root.xpath('//app/b/cell/text()')
		self.c = root.xpath('//app/c/cell/text()')
		self.d = root.xpath('//app/d/cell/text()')
		Q1r = root.xpath('//app/q/firstrow/cell/text()')
		Q2r = root.xpath('//app/q/secondrow/cell/text()')
		Q3r = root.xpath('//app/q/thirdrow/cell/text()')
		Q4r = root.xpath('//app/q/fourthrow/cell/text()')
		self.Q = [Q1r,Q2r,Q3r,Q4r]
		self.m = root.xpath('//app/rlen/cell/text()')
		self.r = root.xpath('//app/r/cell/text()')
		A1r = root.xpath('//app/A/firstrow/cell/text()')
		A2r = root.xpath('//app/A/secondrow/cell/text()')
		A3r = root.xpath('//app/A/thirdrow/cell/text()')
		A4r = root.xpath('//app/A/fourthrow/cell/text()')
		self.A = [A1r,A2r,A3r,A4r]
		B1r = root.xpath('//app/B/firstrow/cell/text()')
		B2r = root.xpath('//app/B/secondrow/cell/text()')
		B3r = root.xpath('//app/B/thirdrow/cell/text()')
		B4r = root.xpath('//app/B/fourthrow/cell/text()')
		self.B = [B1r,B2r,B3r,B4r]
		self.max1 = root.xpath('//app/max1/cell/text()') #0: maximization prob, 1: minimization prob
		self.max2 = root.xpath('//app/max2/cell/text()')
		#conversion to int

		self.xlength = int(self.xlength[0])
		self.ylength = int(self.ylength[0])
		self.m = int(self.m[0])
		for i in range(len(self.a)):
			self.a[i] = int(self.a[i])
		for i in range(len(self.b)):
			self.b[i] = int(self.b[i])
		for i in range(len(self.c)):
			self.c[i] = int(self.c[i])
		for i in range(len(self.d)):
			self.d[i] = int(self.d[i])
		for i in range(len(self.Q)):
			for j in range(len(self.Q[i])):
				self.Q[i][j] = int(self.Q[i][j])
		for i in range(len(self.r)):
			self.r[i] = int(self.r[i])
		for i in range(len(self.A)):
			for j in range(len(self.A[i])):
				self.A[i][j] = int(self.A[i][j])
		for i in range(len(self.B)):
			for j in range(len(self.B[i])):
				self.B[i][j] = int(self.B[i][j])
		self.a = np.array(self.a,dtype = np.int8)
		self.b = np.array(self.b,dtype = np.int8)
		self.c = np.array(self.c, dtype = np.int8)
		self.d = np.array(self.d, dtype = np.int8)
		self.r = np.array(self.r, dtype = np.int8)
		self.A = np.array(self.A, dtype = np.int8)
		self.B = np.array(self.B, dtype = np.int8)
		self.Q = np.array(self.Q, dtype = np.int8)
		self.max1 = int(self.max1[0])
		self.max2 = int(self.max2[0])
		self.ubound = 50 #max value for variables. It can be changed

	def get_feasible(self,chrm): #input: chromosome
		x = np.array([],dtype = np.int8)
		uzeros = np.array([],dtype=np.int8) #0: the corresponding u value must be 0. 1: the corresponding u value must be >= 0
		wzeros = np.array([],dtype=np.int8) #same
		vzeros = np.array([],dtype=np.int8) #same
		yzeros = np.array([],dtype=np.int8) #same
		for i in range(self.m):
			uzeros = np.append(uzeros,[0 if chrm[i]==0 else 1])
			wzeros = np.append(wzeros,[1 if chrm[i]==0 else 0])
		for i in range(self.ylength):
			vzeros = np.append(vzeros,[0 if chrm[self.m+i]==0 else 1])
			yzeros = np.append(yzeros,[1 if chrm[self.m+i]==0 else 0])

		for i in range(self.xlength):
			x = np.append(x,[random.randint(0,self.ubound)])
		#print("chromosome:",chrm)
		y,z = self.calculate(x,uzeros,wzeros,vzeros,yzeros)
		if isinstance(y,(list,pd.core.series.Series,np.ndarray)):
			ynew = yzeros.copy()
			tmp = 0
			for i in range(len(ynew)):
				if ynew[i] == 1:
					ynew[i] *= y[tmp]
					tmp += 1
			y = ynew.copy()
		#print("result:",(x,y,z))
		return x,y,z

	def calculate(self,x,uzeros,wzeros,vzeros,yzeros):
		u = np.zeros(self.count_ones(uzeros),dtype = np.int8)
		w = np.zeros(self.count_ones(wzeros),dtype = np.int8)
		v = np.zeros(self.count_ones(vzeros),dtype = np.int8)
		y = np.zeros(self.count_ones(yzeros),dtype = np.int8)
		bfirst = self.delete_l(self.b,yzeros)
		Bfirst = self.delete_mcol(self.B,yzeros)
		Bsecond = self.delete_mrow(self.B,uzeros)
		Q0 = self.get_submatr(self.Q,len(self.Q)-self.ylength,len(self.Q)-self.ylength,self.ylength,self.ylength)
		Q1 = self.get_submatr(self.Q,self.xlength,0,self.ylength,self.xlength)
		Q0 = self.delete_mcol(Q0,yzeros)
		#print("----------------------- test ----------------------------")
		"""
		print("uzeros",uzeros)
		print("wzeros",wzeros)
		print("vzeros",vzeros)
		print("yzeros",yzeros)
		print("y",y)
		print("u",u)
		print("w",w)
		print("v",v)
		print("x",x)
		print("r",self.r)
		print("a",self.a)
		print("A",self.A)
		print("b",self.b)
		print("B",self.B)
		print("bfirst",bfirst)
		print("Bfirst",Bfirst)
		print("Bsecond",Bsecond)
		print("Q",self.Q)
		print("Q0",Q0)
		print("Q1",Q1)
		"""
		y,z = self.simplex(x,u,w,v,y,bfirst,Bfirst,Bsecond,Q0,Q1)
		return y,z

	def next(self,u,w,v):
		found = False
		tmp = np.concatenate((u,w,v))
		for i in range(len(tmp)):
			if tmp[-i-1] == 0:
				tmp[-i-1] = 1
				for j in range(i):
					tmp[-j-1] = 0
				break
		u = tmp[:self.m]
		w = tmp[self.m:self.m+self.m]
		v = tmp[self.m+self.m:]
		return u,w,v

	def get_submatr(self,m,x,y,length1,length2): #get quadratic submatr starting from (x,y) and spanning for length
		mat = np.array([],dtype = np.int8)
		for i in range(length1):
			tmp = np.array([],dtype = np.int8)
			for j in range(length2):
				tmp = np.append(tmp,m[x+i][y+j])
			mat = np.append(mat,tmp)
		mat = mat.reshape(length1,-1)
		return mat

	def delete_l(self,a,l): #deletes elements based on if the corresponding value inside list l is 1 or 0 (0 -> delete)
		if len(a) != len(l):
			return -1 #len must be equal
		b = np.array([],dtype = np.int8)
		for i in range(len(a)):
			if l[i] == 1:
				b = np.append(b,[a[i]])
		return b

	def delete_mcol(self,m,l): #deletes a column based on if the corresponding value inside l is 1 or 0 (0 -> delete)
		if len(m) > 0 and len(m[0]) != len(l):
			return -1 #len must be equal
		#b = np.empty([len(m),len(m[0])])
		b = np.array([],dtype = np.int8)
		c=0
		for i in range(len(m[0])):
			if l[i] == 1:
				b = np.append(b,m[:,i])
				c += 1
		if b.size != 0:
			b = b.reshape(c,-1)
			b = np.transpose(b)
		return b

	def delete_mrow(self,m,l): #deletes a row based on if the corresponding value inside l is 1 or 0 (0 -> delete)
		if len(m) != len(l):
			return -1
		b = np.array([], dtype = np.int8)
		c = 0
		for i in range(len(m)):
			if l[i] == 1:
				b = np.append(b,m[i,:])
				c += 1
		if b.size != 0:
			b = b.reshape(c,-1)
			b = np.transpose(b)
		return b


	def count_ones(self,l): #count number of ones in a list
		c = 0
		for i in range(len(l)):
			if l[i] == 1:
				c += 1
		return c

	def end_comp(self,u,w,v): #when all slack variables reach a maximum, end the computation to avoid infinite computation
		end = True
		for i in range(len(u)):
			if u[i] != self.ubound:
				end = False
		for i in range(len(w)):
			if w[i] != self.ubound:
				end = False
		for i in range(len(v)):
			if v[i] != self.ubound:
				end = False
		return end

	def simplex(self,x,u,w,v,y,bfirst,Bfirst,Bsecond,Q0,Q1):
		tableaut = self.create_tableaut(x,u,w,v,y,bfirst,Bfirst,Bsecond,Q0,Q1)
		pivot = 0
		#print("simplex")
		while self.pivot_col(tableaut) != -1:
			#print("Tableaut:")
			#print(tableaut)
			piv_col = self.pivot_col(tableaut) #piv is the most negative entry
			ratio = self.get_ratio(tableaut,piv_col)
			pivot = self.get_pivot(ratio)
			#print("pivot: (",pivot,",",piv_col,")")
			if pivot == -1:
				break
			num = tableaut[pivot][piv_col]
			for i in range(len(tableaut[0])):
				tableaut[pivot][i] /= num #making pivot element equal to 1
			for i in range(len(tableaut)):
				n = tableaut[i][piv_col].copy()
				for j in range(len(tableaut[i])):
					if n != 0 and i != pivot:
						tableaut[i][j] -= n*tableaut[pivot][j]
		#print("Tableaut end")
		#print(tableaut)
		if pivot == -1:
			return -1,-1
		else:
			return self.get_y(tableaut,y), tableaut[-1][-1] #solution y and value of the function (tableaut[-1][-1])

	def get_y(self,tableaut,y):
		for i in range(len(y)):
			if not self.basic(tableaut[:,i]):
				y[i] = 0
			else:
				tmp = tableaut[:,i]
				for j in range(len(tmp)):
					if tmp[j] == 1:
						y[i] = tableaut[j][-1]
		return y

	def basic(self,l):
		c = 0
		for o in l:
			if o == 1:
				c += 1
			elif o != 0 and o != 1:
				c = -1 #non basic
				break
		return c == 1

	def get_pivot(self,ratio): #get lowest non negative ratio
		min = float("inf")
		pos = -1
		for i in range(len(ratio)):
			if min > ratio[i] and ratio[i] > 0:
				min = ratio[i]
				pos = i
		return pos

	def pivot_col(self,tableaut): #here the last value of the last row (free variable) is not considered. TODO: check this
		min = 0
		pos = -1
		for i in range(len(tableaut[-1])-1):
			if tableaut[-1][i] < min:
				min = tableaut[-1][i]
				pos = i
		return pos

	def get_ratio(self,tableaut,piv_col): #returns a list containing the ratios of the pivot column
		num = len(tableaut)
		l = np.array([])
		for i in range(num-1):
			if tableaut[i][piv_col] == 0 or tableaut[i][-1] == 0:
				l = np.append(l,[0])
			else:
				l = np.append(l,[tableaut[i][-1]/tableaut[i][piv_col]])
		#print(l)
		return l


	def create_tableaut(self,x,u,w,v,y,bfirst,Bfirst,Bsecond,Q0,Q1):
		#y1 y2 ... yn2 u1 u2 .. um w1 w2 .. wm v1 v2 .. vn2 z   rem
		a = self.a
		if self.max1 == 1: #turn minimization prob to maximization prob
			a = -a
			bfirst = -bfirst
		tableaut = np.array([],dtype = np.int8)
		Ax = np.matmul(self.A,x)
		rterm = self.r - Ax
		rterm2 = -self.d - 2*np.matmul(Q1,x)
		for i in range(len(Bfirst)):
			tmp = np.append([],Bfirst[i,:])
			for j in range(len(u)):
				tmp = np.append(tmp,[0])
			for j in range(len(w)):
				tmp = np.append(tmp,[1])
			for j in range(len(v)):
				tmp = np.append(tmp,[0])
			tmp = np.append(tmp,[0]) #z
			tmp = np.append(tmp,rterm[i]) #result
			tableaut = np.append(tableaut,tmp)

		for i in range(len(Q0)):
			tmp = np.append([],Q0[i,:])
			tmp = 2*tmp
			print("tmp",tmp)
			print("Bsecond",Bsecond)
			if len(Bsecond) != 0:
				tmp = np.append(tmp,-Bsecond[i,:]) #?
			for j in range(len(w)):
				tmp = np.append(tmp,[0])
			for j in range(len(v)):
				if j<i:
					tmp = np.append(tmp,[0])
				elif j==i:
					tmp = np.append(tmp,[1])
				else:
					tmp = np.append(tmp,[0])
			tmp = np.append(tmp,[0]) #z
			tmp = np.append(tmp,rterm2[i])
			tableaut = np.append(tableaut,tmp)

		#print(tableaut.reshape([-1,len(y)+len(u)+len(w)+len(v)+2]))

		mb = -bfirst

		tmp = np.append([],mb)
		for i in range(len(u)+len(w)+len(v)):
			tmp = np.append(tmp,[0])
		tmp = np.append(tmp,[1]) #z
		ax = np.matmul(np.transpose(a),x)
		tmp = np.append(tmp,ax)
		tableaut = np.append(tableaut,tmp)
		tableaut = tableaut.reshape([-1,len(y)+len(u)+len(w)+len(v)+2])
		return tableaut
