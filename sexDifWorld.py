import numpy as np
import copy
import pdb
import Learners
import matplotlib.pyplot as plt
import random
from scipy.stats.stats import pearsonr
from scipy.stats import norm
from scipy.optimize import fsolve
from scipy.stats import binom
import importlib
import logging
importlib.reload(Learners)

class world(object):
	def __init__(self,
			numActs,
			abilityDistribution,
			beliefDif,
			popSize,
			deathRate,
			popMean=100.0, 
			popScale=20.0,
			kindDif=15.0,
			priors = None,
			popLearns = 0,
			# Lets us specify a new ability distro to occur after a given number of timesteps
			abilityDistroUpdateTime = None,
			abilityDistroUpdate = None,
			# Lets us specify a 'allowlist' of events (not yet fully implemented)
			allowLists = None,
			actParents = None,
			learnerAdditions = dict()): # [prob0, prob1, probEven] 
			  
		self.abilityDistroUpdateTime = abilityDistroUpdateTime
		self.abilityDistroUpdate = abilityDistroUpdate
		if allowLists is not None:
			self.allowListStore = allowLists
		else:
			self.allowListStore = dict()
		self.allowList = np.ones(numActs, dtype=bool)
		if actParents is not None:
			self.actParents = actParents
		else:
			self.actParents = [-1] * numActs

		self.learnerAdditions = learnerAdditions
		
		# underlying economy and aptitude distribution
		self.numActs = numActs
		self.abilityDistribution = abilityDistribution
		[self.num0Acts, self.num1Acts, self.numSameActs] = np.random.multinomial(self.numActs,abilityDistribution)
		self.kindDif = kindDif
		self.beliefDif = beliefDif
		
		self.popMean = popMean
		self.popScale = popScale
		
		#demographics
		
		self.deathRate = deathRate
		self.popLearns = popLearns
		
		# records of things		
		self.moneySegHist = []
		self.moneyTotalHist = []
		self.priceHist = []
		self.priceHist.append(np.ones(self.numActs))
		self.segregationHist = []
		self.weightedSegregationHist = []
		
		# search threshold
		self.threshold, self.explorePrediction = self.numOpt()
		print(self.threshold)
		print(self.explorePrediction)
		
		# priors for making inferences
		if priors == None:
			[self.priorProb0, self.priorProb1, self.priorProbEven] = abilityDistribution
		else:
			self.priorProb0 = priors[0]
			self.priorProb1 = priors[1]
			self.priorProbEven = priors[2]
			
		#populate the world
		self.learners = []
		self.learner = Learners.learner
		
		
			
		#initialize the worker allocation
		self.kindEstimates = np.vstack([np.array([self.popMean*1.0]*self.numActs), np.array([self.popMean*1.0]*self.numActs)])
		self.actCounts   = np.zeros((2, self.numActs))
		self.actExploits = np.zeros((2, self.numActs))
		self.actAttempts = np.zeros((2, self.numActs))
		
		# Filled in in 'addLearners'
		self.production = np.zeros(self.numActs)
		self.typeProduction = [np.zeros(self.numActs), np.ones(self.numActs)]
		# self.production  = np.ones(self.numActs) * popSize # everybody makes one of everything for free
		# self.typeProduction = [np.ones(self.numActs) * popSize/2, np.ones(self.numActs) * popSize/2]
		self.explored    = 0.0

		self.popSize = 0 # This is updated by 'addLearners'
		self.addLearners(popSize)
		self.priceHist.append((1.0 / self.production) * self.priceNorm)
	
	def addLearners(self, numLearners):
		logging.info("Adding %d learners" % numLearners)
		oldlearners = len(self.learners)

		self.production  += np.ones(self.numActs) * numLearners # everybody makes one of everything for free
		self.typeProduction += [np.ones(self.numActs) * numLearners/2, np.ones(self.numActs) * numLearners/2]
		self.popSize += numLearners

		self.threshold, self.explorePrediction = self.numOpt()

		# price normalization term
		self.priceNorm = self.popSize * ( 1.0 + (self.threshold / self.numActs))
	

		for ii in range(np.int(numLearners/2)):
			#kind 0
			aptitudes = np.zeros(self.numActs)
			aptitudes[0:self.num0Acts] = np.random.normal(loc=self.popMean+self.kindDif, scale=self.popScale, size=self.num0Acts)
			aptitudes[self.num0Acts:self.num0Acts+self.num1Acts] = np.random.normal(loc=self.popMean-self.kindDif, scale=self.popScale, size=self.num1Acts)
			aptitudes[self.num0Acts+self.num1Acts:self.numActs] = np.random.normal(loc=self.popMean, scale=self.popScale, size=self.numSameActs)
			aptitudes[aptitudes < 0] = 0
			
			self.learners.append(self.learner(aptitudes = aptitudes, kind = 0, learns = self.popLearns))
			
			#kind 1
			aptitudes = np.zeros(self.numActs)
			aptitudes[0:self.num0Acts] = np.random.normal(loc=self.popMean-self.kindDif, scale=self.popScale, size=self.num0Acts)
			aptitudes[self.num0Acts:self.num0Acts+self.num1Acts] = np.random.normal(loc=self.popMean+self.kindDif, scale=self.popScale, size=self.num1Acts)
			aptitudes[self.num0Acts+self.num1Acts:self.numActs] = np.random.normal(loc=self.popMean, scale=self.popScale, size=self.numSameActs)
			aptitudes[aptitudes < 0] = 0
			
			self.learners.append(self.learner(aptitudes = aptitudes, kind = 1, learns = self.popLearns))

		for ll in self.learners[oldlearners:]:
			if ll.kind == 0:
				act, amount, known, explored = ll.choice(self.priceHist[-1], self.kindEstimates, self.allowList, threshold = self.threshold)
				self.actCounts[0, act] = self.actCounts[0, act] + 1
				if explored == 0.0:
					# acts explored this round are not used for estimates
					self.actExploits[0, act] = self.actExploits[0, act] + 1
				self.actAttempts[0,:] = self.actAttempts[0,:] + known
			elif ll.kind == 1:
				act, amount, known, explored = ll.choice(self.priceHist[-1], self.kindEstimates, self.allowList, threshold = self.threshold)
				self.actCounts[1, act] = self.actCounts[1, act] + 1
				if explored == 0.0:
					# acts explored this round are not used for estimates
					self.actExploits[1, act] = self.actExploits[1, act] + 1
				self.actAttempts[1, :] = self.actAttempts[1,:] + known
				
			self.production[act] = self.production[act] + amount
			self.typeProduction[ll.kind][act] += amount 
			self.explored = self.explored + explored


		self.makeKindEstimates()
		self.updatePrices()

	def updatePrices(self):
		self.prices = np.where(self.allowList, (1.0 / self.production) * self.priceNorm, -100000)
		
	def run(self, n=2):
		for pp in range(n):
			logging.info("Round %d" % pp)
			# Update aptitude (if reached correct timestep)
			if self.abilityDistroUpdateTime is not None:
				self.abilityDistroUpdateTime -= 1
				if self.abilityDistroUpdateTime <= 0:
					print("Updating acts:", [self.num0Acts, self.num1Acts, self.numSameActs])
					[self.num0Acts, self.num1Acts, self.numSameActs] = np.random.multinomial(self.numActs,self.abilityDistroUpdate)
					self.abilityDistroUpdateTime = None
					print("Updated to:", [self.num0Acts, self.num1Acts, self.numSameActs])

			if pp in self.allowListStore:
				print("Update allowList at " + str(pp));
				self.allowList = self.allowListStore[pp]
				self.updatePrices()

			# Update learners
			if pp in self.learnerAdditions:
				self.addLearners(self.learnerAdditions[pp])
				self.makeKindEstimates()
				self.updatePrices()

			np.random.shuffle(self.learners)
			# choose a learner at random from the population
			# they will change their activity based on their decision rule
			for ii,ll in enumerate(self.learners):

				#ll = self.learners[ii]
				#undo what they did last time
				act = ll.actHist[-1]; amount = ll.aptitudes[act]; known = (ll.knownAptitudes > -1.0)
				explored = 0.0 
				if ll.state == 'explore': 
					explored = 1.0
			
				# update the records
				self.actCounts[ll.kind, act]  = self.actCounts[ll.kind, act] - 1.0
				if explored == 0.0:
					# acts explored this round are not used for estimates
					self.actExploits[ll.kind, act]  = self.actExploits[ll.kind, act] - 1.0
				self.actAttempts[ll.kind] = self.actAttempts[ll.kind] - known
			
				self.production[act] = self.production[act] - amount
				self.typeProduction[ll.kind][act] -= amount
				self.explored = self.explored - explored
				
				# update the kindEstimate for the act that this learner may just be leaving
				self.updateKindEstimate(act)
				
				# see if they live first, or are replaced by a naive learner
				if np.random.rand() < self.deathRate:
					ll = self.deathBirth(ii,ll)				
				
				#figure out what gets done and update the records
				act, amount, known, explored = ll.choice(self.prices, self.kindEstimates, self.allowList, threshold = self.threshold)
				
				self.actCounts[ll.kind, act] = self.actCounts[ll.kind, act] + 1
				if explored == 0.0:
					# acts explored this round are not used for estimates
					self.actExploits[ll.kind, act] = self.actExploits[ll.kind, act] + 1
				self.actAttempts[ll.kind] = self.actAttempts[ll.kind] + known	
				self.production[act] = self.production[act] + amount
				self.typeProduction[ll.kind][act] += amount
				self.explored = self.explored + explored
				
				# update the prices and the estimates
				self.updateKindEstimate(act)
				self.updatePrices()
				
				#print self.prices
				#print(min(self.prices))
				#print(max(self.prices))
				
			# our segreation index is the absolute deviation from 0.5 averaged over all of the activities 
			# the scaled by 2 so that 1.0 is total segreation and 0.0 is perfect balance... 
			# what noise gives depends on populations size and number of activities 
			# print("actcounts", self.actCounts)
			segregation = 2 * np.mean(np.abs((self.actCounts[0,self.allowList] / (self.actCounts[0,self.allowList] + self.actCounts[1,self.allowList])) - 0.5))
			self.segregationHist.append(segregation)

			weightedSegregation = 2 * np.mean(np.abs((self.actCounts[0,self.allowList] / (self.actCounts[0,self.allowList] + self.actCounts[1,self.allowList])) - 0.5)*(self.actCounts[0,self.allowList]+self.actCounts[1,self.allowList]))
			self.weightedSegregationHist.append(weightedSegregation/(np.sum(self.actCounts[0,self.allowList]+self.actCounts[1,self.allowList])/np.sum(self.allowList)))


			values = [np.sum(self.typeProduction[i] * self.prices) for i in [0,1]]

			self.moneySegHist.append((values[0] - values[1]) / (values[0] * values[1]))
			self.moneyTotalHist.append((values[0] * values[1]))

			self.priceHist.append((1.0 / self.production) * self.priceNorm)
			#print('round: ' + str(pp) + ' explorers: ' + str(self.explored/self.popSize) + ' seg: ' + str (segregation))
				
	def deathBirth(self, ii, ll):
		#get aptitudes for new learner
		if ll.kind == 0:
			aptitudes = np.zeros(self.numActs)
			aptitudes[0:self.num0Acts] = np.random.normal(loc=self.popMean+self.kindDif, scale=self.popScale, size=self.num0Acts)
			aptitudes[self.num0Acts:self.num0Acts+self.num1Acts] = np.random.normal(loc=self.popMean-self.kindDif, scale=self.popScale, size=self.num1Acts)
			aptitudes[self.num0Acts+self.num1Acts:self.numActs] = np.random.normal(loc=self.popMean, scale=self.popScale, size=self.numSameActs)
			aptitudes[aptitudes < 0] = 0
		elif ll.kind == 1:
			aptitudes = np.zeros(self.numActs)
			aptitudes[0:self.num0Acts] = np.random.normal(loc=self.popMean-self.kindDif, scale=self.popScale, size=self.num0Acts)
			aptitudes[self.num0Acts:self.num0Acts+self.num1Acts] = np.random.normal(loc=self.popMean+self.kindDif, scale=self.popScale, size=self.num1Acts)
			aptitudes[self.num0Acts+self.num1Acts:self.numActs] = np.random.normal(loc=self.popMean, scale=self.popScale, size=self.numSameActs)
			aptitudes[aptitudes < 0] = 0
		
		# replace the old learner with the new
		
		self.learners[ii] = self.learner(aptitudes = aptitudes, kind = ll.kind, learns = self.popLearns)
		return self.learners[ii]
		
	def makeKindEstimates(self):
		#social informaiton for next time based on actCounts by kind, attempts by kind, and priors
		if self.priorProbEven == 1:
			self.kindEstimates = np.zeros((2,self.numActs)) + self.popMean
		else:
			for aa in range(self.numActs):
				self.updateKindEstimate(aa)


	def updateKindEstimate(self, act):
		#social informaiton for next time based on actCounts by kind, attempts by kind, and priors
		if self.priorProbEven < 1.0:
			# total number doing this act
			num0Doing = self.actExploits[0,act]
			num0Not   = self.actAttempts[0,act] - self.actExploits[0,act]
			num1Doing = self.actExploits[1,act]
			num1Not   = self.actAttempts[1,act] - self.actExploits[1,act]

			#print("0x", num0Doing, num0Not, num1Doing, num1Not)
			# If we have a parent, and it is currently active
			if self.actParents[act] != -1 and self.allowList[self.actParents[act]]:
				pact = self.actParents[act]
				num0Doing += self.actExploits[0,pact]
				num0Not   += self.actAttempts[0,pact] - self.actExploits[0,pact]
				num1Doing += self.actExploits[1,pact]
				num1Not   += self.actAttempts[1,pact] - self.actExploits[1,pact]

			#print("1x:", num0Doing, num0Not, num1Doing, num1Not)

			assert min(num0Doing, num0Not, num1Doing, num1Not) >= 0
			assert max(num0Doing, num0Not, num1Doing, num1Not) <= len(self.learners)*2
			
			#need to use prices from last round as that's what these decisions were based on
			skillThreshold = self.threshold / self.priceHist[-1][act]

			#print("skillThreshold:", skillThreshold)
			# prob an agent's skill is below the required threshold given they are favoured at this activity
			Fplus = norm.cdf(skillThreshold, loc = self.popMean + self.beliefDif, scale = self.popScale)
			# prob an agent's skill is below the required threshold given they are unfavoured at this activity
			Fminus = norm.cdf(skillThreshold, loc = self.popMean - self.beliefDif, scale = self.popScale)
			# prob an agent's skill is below the required threshold given they are neither favoured nor unfavoured
			Feven = norm.cdf(skillThreshold, loc = self.popMean, scale = self.popScale)
			
			##print("Fplus:", Fplus, "Fminus:", Fminus, "Feven:", Feven)

			# Escape if all equal
			if Fplus==[1.0] or Fminus==[1.0] or Feven==[1.0]:
				self.kindEstimates[:,act] = np.zeros(2) + self.popMean
				return
			
			# given events, prob that this activitity favours 0 types
			A0 = self.priorProb0 * np.exp(num0Not * np.log(Fplus) + num0Doing * np.log(1-Fplus) + num1Not * np.log(Fminus) +  num1Doing * np.log(1-Fminus))
			# given events, prob that this activity favours 1 types
			A1 = self.priorProb1 * np.exp(num0Not * np.log(Fminus) + num0Doing * np.log(1-Fminus) + num1Not * np.log(Fplus) + num1Doing * np.log(1-Fplus))
			# given events, prob that this activity favours neither type
			A2 = self.priorProbEven * np.exp(num0Not * np.log(Feven) + num0Doing * np.log((1-Feven)) + num1Not * np.log(Feven) + num1Doing * np.log(1-Feven))
			
			#print("A0:", A0, "A1:", A1, "A2:", A2)
			# If we can deduce nothing, set everything to equal
			if A0 + A1 + A2 == 0:
				A0 = self.priorProb0
				A1 = self.priorProb1
				A2 = self.priorProbEven
			normterm = A0 + A1 + A2
			if not np.all(np.isfinite(np.array([A0,A1,A2]))):
				print(A0)
				print(A1)
				print(A2)	
				pdb.set_trace()
			try:	
				
				A0 = A0/normterm
				A1 = A1/normterm
				A2 = A2/normterm
				
				self.kindEstimates[0,act] = ((self.popMean+self.beliefDif) * A0 + 
							  (self.popMean-self.beliefDif) * A1 + 
							  (self.popMean) * A2)
				self.kindEstimates[1,act] = ((self.popMean-self.beliefDif) * A0 + 
							  (self.popMean+self.beliefDif) * A1 + 
							  (self.popMean) * A2)
			except:
				print(A0)
				print(A1)
				print(A2)	
				pdb.set_trace()
		else:
			self.kindEstimates[:,act] = np.zeros(2) + self.popMean


	def countsByKind(self):
		
		Counts0 = np.zeros(self.numActs)
		Counts1 = np.zeros(self.numActs)
		for ll in self.learners:
			if ll.kind == 0:
				Counts0 = Counts0 + ll.actCounts/ll.age/(self.popSize/2.0)
			elif ll.kind == 1:
				Counts1 = Counts1 + ll.actCounts/ll.age/(self.popSize/2.0)
		plt.scatter(range(1,self.numActs+1), Counts0, c = 'blue')
		plt.scatter(range(1,self.numActs+1), Counts1, c = 'red')
		plt.xlabel('Act type')
		plt.ylabel('Average number times performed per period by each type')
		plt.show()
	
	def countsByApt(self):
		x = []
		y = []
		for ll in self.learners:
			for aa in range(self.numActs):
				x.append(ll.aptitudes[aa])
				y.append(ll.actCounts[aa]/ll.age)
		plt.scatter(x,y)
		plt.xlabel('Aptitude')
		plt.ylabel('Total number times performed normalized by age')
		print(pearsonr(x,y))
		plt.show()
		
	def numOpt(self):
		# We will need the expected payoff of an activity given that it is above a certain threshold T 
		# which in turn the cdf and pdf of the ability distribution evaluated at the threshold value
		# note that the actual ability distribution for a randomly sampled activity is a composite of three different normal distributions
		
		PDFOnCDFPlus  = lambda T: norm.pdf(T, loc = (self.popMean + self.kindDif), scale = self.popScale) / (1.0 - norm.cdf(T, loc = (self.popMean + self.kindDif), scale = self.popScale))
		PDFOnCDFMinus = lambda T: norm.pdf(T, loc = (self.popMean - self.kindDif), scale = self.popScale) / (1.0 - norm.cdf(T, loc = (self.popMean - self.kindDif), scale = self.popScale))
		PDFOnCDFEven  = lambda T: norm.pdf(T, loc = self.popMean, scale = self.popScale) / (1.0 - norm.cdf(T, loc = self.popMean, scale = self.popScale))
		
		EAbove = lambda T: self.popMean + self.popScale**2 * np.dot(self.abilityDistribution, np.array([PDFOnCDFPlus(T), PDFOnCDFMinus(T), PDFOnCDFEven(T)]))
		PBelow = lambda T: np.dot(self.abilityDistribution, np.array([norm.cdf(T, loc=(self.popMean + self.kindDif), scale=self.popScale), norm.cdf(T, loc=(self.popMean-self.kindDif), scale=self.popScale), norm.cdf(T, loc=self.popMean, scale = self.popScale)]))
		
		# This equation come from appendix the optimal threshold is the unique solution to this equation,
		# which is the threshold where the value of observing is equal to the value of exploitin
		dummy = lambda T: T*(1-(1-self.deathRate)*PBelow(T)) - self.popMean * self.deathRate  - (1.0-self.deathRate) * (1.0-PBelow(T)) * (EAbove(T))
		maxX = fsolve(dummy, self.popMean+self.popScale)
		
		explorePrediction = self.deathRate / (self.deathRate + (1-PBelow(maxX)))
		
		return maxX, explorePrediction

