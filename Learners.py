import logging
import numpy as np
from scipy.stats import norm
import pdb

NAME = 0

THRESHOLD_CHANGE = 1.0

def set_threshold_change(t):
	global THRESHOLD_CHANGE
	THRESHOLD_CHANGE = t

PERFECT_KNOWLEDGE = False

def set_perfect_knowledge(p):
	global PERFECT_KNOWLEDGE
	PERFECT_KNOWLEDGE = p

"""
These learners use counts to estimate which activity they will be best suited to,
not the actual production levels of those engaged in an activity.
They set their utility threshold using the optimal solo search threshold as a proxy,
this is a good proxy under the assumption that prices are stable, equal, and roughly equal to 1.

Using this utility threshold, they infer, in a Bayesian way (with some strong assumptions)
how likely their type is to be favoured at a given task
The strong assumptions are
1) They know the distribution of aptitudes over activities for each of type
2) They know the proportion of types currently engaged in each activity.
3) They know the proportion of types that have ever tried each activity.

In our simple model of ability differences there are three classes of activity.
(0) Activities favoured by type zero so that type zeros have ability distribution Fplus and
type ones have ability distribution Fminus.
(1) Activities favoured by type one so that type ones have ability distribution Fplus and
type zeros have ability distribution Fminus.
(2) Both types share a single ability distribution F(x).

The observed number of type zero individuals engaged in a given activity will be
binomially distributed with
N = number doing the activity
k = probability an agent doing the activity is type 0.
let p = the probability that an agent who has tried the activity (but may or may not be currently doing it) is type 0

k0 = (prob old * (p*Fplus)/(p*Fplus  + (1-p)*Fminus) + prob new * p) in case (0) of type zero favoured
k1 = (prob old * (p*Fminus)/(p*Fminus + (1-p)*Fplus) + prob new * p) in case (1) of type zero favoured
k2 = p in the case (2) of equal aptitudes

We will assume for simplicity though that prob old ~ 1 and that prob new ~ 0

Prob(Activity is type i | observed proportion x/n) =
Prob( Observing that proportion | Activity is type i) * Prob(Activity is type i) / Normalization Term

This computation is done once for the whole population and then passed to the learners, so that each round
they base their decision on recieved
- prices
- wealth/utility thresholds
- expected ability at a given activity derived from
probabilities of different activities favouring a given type which is in turn infered from
proporiton of each type engaged in that activity, and empirical encounter rates, and prior beliefs about the ability distribution.

Thus differences in decisions arise only from the personal knowledge agents have of their own abilities based on experience.
Thus two completely naive individuals of the same type will make identical decisions.
"""
class learner(object):
	def __init__(self, *, aptitudes,
			kind,
			learns = 0):

		global NAME
		self.name = hex(NAME) + "-" + str(kind)
		NAME += 1
		logging.info("Learner %s is born" % (self.name))
		self.numActs = len(aptitudes)
		self.kind = kind
		self.learns = learns

		self.resources = np.ones(self.numActs)

		self.actHist = []
		self.payHist = []

		self.actCounts = np.zeros(self.numActs)

		self.age = 1

		if aptitudes is None:
			print('you blew it')
		else:
			self.aptitudes = aptitudes

		if PERFECT_KNOWLEDGE:
			self.knownAptitudes = aptitudes
		else:
			#You learn your aptitudes, but you 'know' the value of exploring
			self.knownAptitudes = -np.ones(self.numActs)
		self.state = 'explore'

	def choice(self, prices, kindEstimates, allowList, *, threshold):
		threshold = threshold * THRESHOLD_CHANGE

		known   = (self.knownAptitudes  > -1 & allowList)
		unknown = (self.knownAptitudes == -1 & allowList)


		def getMax(est):
			e = est[allowList]
			m = np.max(e)
			return np.where( (est==m) & allowList )[0]

		if len(self.actHist) == 0:
			utEst = kindEstimates[self.kind] * prices
			#print("!A", utEst, np.where(utEst == np.max(utEst)))
			todaysActivity = np.random.choice(getMax(utEst))
			self.state = 'explore'
			logging.info("Learner %s is starting: %d" % (self.name, todaysActivity))
		elif sum(known) == self.numActs:
			utEst = self.knownAptitudes * prices
			#print("!B", utEst, np.where(utEst == np.max(utEst)))
			todaysActivity = np.random.choice(np.where(utEst == np.max(utEst))[0])
			self.state = 'exploit'
			#print('all out of acts')
		else:
			lastAct = self.actHist[-1]
			utEst   = np.zeros(self.numActs)
			utEst[known] = self.knownAptitudes[known] * prices[known]

			# do we know any activities over the threshold, if so do the best of those
			if np.max(utEst) >= threshold and len(self.actHist) > self.learns:
				#print("!C", utEst, np.where(utEst == np.max(utEst)))
				todaysActivity = np.random.choice(getMax(utEst))
				if self.state != 'exploit':
					logging.info("Learner %s is moving to exploit: %d" % (self.name, todaysActivity))
				self.state = 'exploit'

			else:
				#nothing known is above the threshold
				utEst = np.zeros(self.numActs)
				utEst[unknown] = kindEstimates[self.kind,unknown] * prices[unknown]
				estsum = np.sum(utEst)
				if estsum > 0:
					probs = utEst/estsum
				else:
					probs = np.ones(len(utEst))/len(utEst)
				#print("!D", utEst, np.where(utEst == np.max(utEst)))
				if np.max(utEst) < 0:
					utEst = self.knownAptitudes * prices
					todaysActivity = np.random.choice(getMax(utEst))
					self.state = 'exploit'
				else:
					todaysActivity = np.random.choice(getMax(utEst))
					logging.info("Learner %s is exploring: %d" % (self.name, todaysActivity))
					self.state = 'explore'
		
		#get the production, update the estimate, update the action counts
		production = self.aptitudes[todaysActivity]
		self.knownAptitudes[todaysActivity] = production
		self.actCounts[todaysActivity] = self.actCounts[todaysActivity] + 1
		self.actHist.append(todaysActivity)

		if self.state == 'explore':
			explored = 1.0
		else:
			explored = 0.0

		self.age = self.age + 1.0

		# Update known
		known = (self.knownAptitudes  > -1)

		return todaysActivity, production, known, explored
