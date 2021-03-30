from genetic import Genetic
import numpy as np
import lxml.etree

class Tester:
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

		self.genetic = Genetic()
		print("Finished initialization process...")

	def __init__(self,path): #path to xml file
		#this part works only with a properly defined xml file passed as input. It is the constructor to use for testing
		with open(path) as f:
			lines = f.readlines()
		lines = '\n'.join(lines)
		root = lxml.etree.fromstring(lines)

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
		B1r = root.xpath('//app/q/firstrow/cell/text()')
		B2r = root.xpath('//app/q/secondrow/cell/text()')
		B3r = root.xpath('//app/q/thirdrow/cell/text()')
		B4r = root.xpath('//app/q/fourthrow/cell/text()')
		self.B = [B1r,B2r,B3r,B4r]
		pop_size = root.xpath('//app/popsize/cell/text()')
		crossprob = root.xpath('//app/crossprob/cell/text()')
		mutationprob = root.xpath('//app/mutationprob/cell/text()')
		maxgeneration = root.xpath('//app/maxgeneration/cell/text()')

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

		#define genetic
		self.genetic = Genetic(int(pop_size[0]),float(crossprob[0]),float(mutationprob[0]),int(maxgeneration[0]))
