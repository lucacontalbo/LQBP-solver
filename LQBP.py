import numpy as np
import lxml.etree

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
		self.ubound = 1000

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
		self.max1 = root.xpath('//app/max1/cell/text()') #1: maximization prob, 2: minimization prob
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
		self.a = np.array(self.a)
		self.b = np.array(self.b)
		self.c = np.array(self.c)
		self.d = np.array(self.d)
		self.r = np.array(self.r)
		self.A = np.array(self.A)
		self.B = np.array(self.B)
		self.Q = np.array(self.Q)
		self.max1 = int(self.max1)
		self.max2 = int(self.max2)
		self.ubound = 1000 #max value for variables. It can be changed

	def get_feasible(self,chrm): #input: chromosome
		x = np.array([])
		uzeros = np.array([]) #0: the corresponding u value must be 0. 1: the corresponding u value must be >= 0
		wzeros = np.array([]) #same
		vzeros = np.array([]) #same
		yzeros = np.array([]) #same
		for i in range(self.m):
			uzeros = np.append(uzeros,[0 if chrm[i]==0 else 1])
			wzeros = np.append(wzeros,[1 if chrm[i]==0 else 0])
		for i in range(self.ylength):
			vzeros = np.append(vzeros,[0 if chrm[self.m+i]==0 else 1])
			yzeros = np.append(yzeros,[1 if chrm[self.m+i]==0 else 0])

		for i in range(self.xlength):
			x = np.append(x,[random.randint(0,self.ubound)])

		x,y,u,w,v = calculate(x,uzeros,wzeros,vzeros,yzeros)
		#TODO: parse y',u',w',v' in y,u,w,v

	def calculate(self,x,uzeros,wzeros,vzeros,yzeros):
		u = np.zeros(self.count_ones(uzeros))
		w = np.zeros(self.count_ones(wzeros))
		v = np.zeros(self.count_ones(vzeros))
		y = np.array(self.count_ones(yzeros))
		bfirst = self.delete_l(self.b,yzeros)
		Bfirst = self.delete_mcol(self.B,yzeros)
		Bsecond = self.delete_mrow(self.B,uzeros)
		Q0 = self.get_submatr(self.Q,len(self.Q)-self.ylength,len(self.Q)-self.ylength,self.ylength,self.ylength)
		Q1 = self.get_submatr(self.Q,self.xlength,0,self.ylength,self.xlength)
		Q0 = self.delete_mcol(Q0,yzeros)
		while(!self.end_comp(u,w,v)):
			val = self.simplex(x,u,w,v,y,bfirst,Bfirst,Bsecond,Q0,Q1) #TODO
			if val != -1:
				break
			u,w,v = self.next(u,w,v)
		y = val
		return x,y,u,w,v

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
		mat = np.array([])
		for i in range(length1):
			tmp = np.array([])
			for j in range(length2):
				tmp = np.append(tmp,m[x+i][y+j])
			mat = np.append(mat,tmp)
		return mat

	def delete_l(self,a,l): #deletes a elements based on if the corresponding value inside list l is 1 or 0 (0 -> delete)
		if len(a) != len(l):
			return -1 #len must be equal
		b = np.array([])
		for i in range(len(a)):
			if l[i] == 1:
				b = np.append(b,[a[i]])
		return b

	def delete_mcol(self,m,l): #deletes a column based on if the corresponding value inside l is 1 or 0 (0 -> delete)
		if len(m) > 0 and len(m[0]) != len(l):
			return -1 #len must be equal
		b = np.array([])
		for i in range(len(m[0])):
			if l[i] == 1:
				b = np.append(b,m[:,i],axis=1)
		return b

	def delete_mrow(self,m,l): #deletes a row based on if the corresponding value inside l is 1 or 0 (0 -> delete)
		if len(m) != len(l):
			return -1
		b = np.array([])
		for i in range(len(m)):
			if l[i] == 1:
				b = np.append(b,m[i,:])
		return b


	def count_ones(self,l): #count number of ones in a list
		c = 0
		for i in range(len(l)):
			if l[i] == 1:
				c += 1
		return c

	def end_comp(self,u,w,v): #when all slack variables reach a maximum, end the computation to avoid infinite computation
		end = True
		for i in range(self.m):
			if u[i] != self.ubound or w[i] != self.ubound:
				end = False
		for i in range(self.ylength):
			if v[i] != self.ubound:
				end = False
		return end
