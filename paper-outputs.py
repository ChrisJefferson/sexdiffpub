import logging
import matplotlib.pyplot as plt
import numpy as np
import sexDifWorld
import Learners
import sys

import argparse
import json

from multiprocessing import Pool

parser = argparse.ArgumentParser(description='SexDiffWorld')

parser.add_argument('--fast', action='store_true', help='Run a smaller, faster model')

parser.add_argument('--priortype', type=int, help="prior type")
parser.add_argument('--model', type=str, help="model to run")
parser.add_argument('--allowlists', type=int, default=0)
parser.add_argument('--actparents', type=int, default=0)
parser.add_argument('--logging', action='store_true')
parser.add_argument('--seed', type=int)
parser.add_argument('--name', type=str)
parser.add_argument('--threshold', type=int, default=100)
parser.add_argument('--outdir', type=str, default='.')
parser.add_argument('--learnerAdditions', type=str, default="{}")
parser.add_argument('--perfect', action='store_true', help='Learners have perfect knowledge')

world = sexDifWorld.world

args = parser.parse_args()

Learners.set_threshold_change(args.threshold/100.0)

if args.perfect:
    Learners.set_perfect_knowledge(True)

if args.seed:
    print("Seeding with", args.seed)
    np.random.seed(args.seed)


if args.logging:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(lineno)d:%(name)s:%(message)s",
    )

# Some constants used in all experiments
if args.fast:
    STEPS = 100
    POPSIZE = 300
    NUMACTS = 24
    DEATHRATE = 0.1
else:
    STEPS = 400
    POPSIZE = 1000
    NUMACTS = 60
    DEATHRATE = 0.05


def runWorld(**kwargs):
    w = world(**kwargs)
    w.run(STEPS)
    #return w.segregationHist
    return {'weightedSegregationHist': w.weightedSegregationHist, 'production': np.sum(w.production)}


def makeGraphs(*,worlds, labels, title, xlabel,  name):
    t = 'weightedSegregationHist' #, 'segregationHist', 'moneyTotalHist', 'moneySegHist']:
    for (w,a) in zip(worlds, labels):
        plt.plot(w[t], label=a)
    
    with open(name + ".csv", "a") as f: 
        f.write(",".join(str(x) for x in w[t]) + "\n")

    with open(name + ".json", "a") as f: 
        f.write(json.dumps(w) + "\n")

    axes = plt.gca()
    axes.set_ylim([0,1])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(t)
    plt.legend()
    plt.savefig(name + ".png")
    plt.close()
        

def model1Func(prop, priors, beliefDif,allowLists,actParents,learnerAdditions):
    return runWorld(abilityDistribution = [prop/2.0, prop/2.0, 1.0 - prop], 
                    popSize = POPSIZE, deathRate = DEATHRATE, priors=priors, beliefDif=beliefDif,allowLists=allowLists,actParents=actParents,learnerAdditions=learnerAdditions,numActs=NUMACTS)

def model1(*,priors, beliefDif,name ,allowLists,actParents,learnerAdditions):
    args = [(0.2, priors, beliefDif,allowLists,actParents,learnerAdditions)]

    with Pool(len(args)) as p:
        worlds = p.starmap(model1Func, args)

    makeGraphs(worlds=worlds, labels=[a[0] for a in args], 
               title=name,
               xlabel='time',
               name = name)   

def model1smallFunc(prop, priors, beliefDif,allowLists,actParents,learnerAdditions):
    return runWorld(abilityDistribution = [prop/2.0, prop/2.0, 1.0 - prop], kindDif=4,
                    popSize = POPSIZE, deathRate = DEATHRATE, priors=priors, beliefDif=beliefDif,allowLists=allowLists,actParents=actParents,learnerAdditions=learnerAdditions,numActs=NUMACTS)

def model1small(*,priors, beliefDif,name ,allowLists,actParents,learnerAdditions):
    args = [(0.2, priors, beliefDif,allowLists,actParents,learnerAdditions)]

    with Pool(len(args)) as p:
        worlds = p.starmap(model1smallFunc, args)

    makeGraphs(worlds=worlds, labels=[a[0] for a in args], 
               title=name,
               xlabel='time',
               name = name)   


def model2Func(priors, beliefDif,allowLists,actParents,learnerAdditions):
    return runWorld(abilityDistribution = [0.0,0.0,1.0], 
                    popSize = POPSIZE, deathRate = DEATHRATE, priors=priors, beliefDif=beliefDif,allowLists=allowLists,actParents=actParents,learnerAdditions=learnerAdditions,numActs=NUMACTS)

def model2(*,priors, beliefDif,name ,allowLists,actParents,learnerAdditions):
    args = [(priors, beliefDif,allowLists,actParents,learnerAdditions)]

    with Pool(len(args)) as p:
        worlds = p.starmap(model2Func, args)

    makeGraphs(worlds=worlds, labels=[a[0] for a in args], 
               title=name,
               xlabel='time',
               name = name)   


def model3bFunc(priors, beliefDif,allowLists,actParents,learnerAdditions):
    return runWorld(abilityDistribution = [0.2, 0.2, 0.6],
             abilityDistroUpdateTime = STEPS/10,
             abilityDistroUpdate = [0.0, 0.0, 1.0],
             popSize = POPSIZE, deathRate = DEATHRATE, priors = priors, beliefDif=beliefDif,allowLists=allowLists,actParents=actParents,learnerAdditions=learnerAdditions,numActs=NUMACTS)

def model3b(*,priors, beliefDif,name ,allowLists,actParents,learnerAdditions):
    args = [(priors, beliefDif,allowLists,actParents,learnerAdditions)]

    with Pool(len(args)) as p:
        worlds = p.starmap(model3bFunc, args)

    makeGraphs(worlds=worlds, labels=[a[0] for a in args], 
               title=name,
               xlabel='time',
               name = name)  


def model4do10Func(priors, beliefDif,allowLists,actParents,learnerAdditions):
    return runWorld(abilityDistribution = [0.2, 0.2, 0.6],
             abilityDistroUpdateTime = STEPS/10,
             abilityDistroUpdate = [0.0, 0.0, 1.0],
             popSize = POPSIZE, deathRate = DEATHRATE, priors = priors, beliefDif=beliefDif,popLearns=10,allowLists=allowLists,actParents=actParents,learnerAdditions=learnerAdditions,numActs=NUMACTS)

def model4do10(*,priors, beliefDif,name ,allowLists,actParents,learnerAdditions):
    args = [(priors, beliefDif,allowLists,actParents,learnerAdditions)]

    with Pool(len(args)) as p:
        worlds = p.starmap(model4do10Func, args)

    makeGraphs(worlds=worlds, labels=[a[0] for a in args], 
               title=name,
               xlabel='time',
               name = name)  

def model4do4Func(priors, beliefDif,allowLists,actParents,learnerAdditions):
    return runWorld(abilityDistribution = [0.2, 0.2, 0.6],
             abilityDistroUpdateTime = STEPS/10,
             abilityDistroUpdate = [0.0, 0.0, 1.0],
             popSize = POPSIZE, deathRate = DEATHRATE, priors = priors, beliefDif=beliefDif,popLearns=4,allowLists=allowLists,actParents=actParents,learnerAdditions=learnerAdditions,numActs=NUMACTS)

def model4do4(*,priors, beliefDif,name ,allowLists,actParents,learnerAdditions):
    args = [(priors, beliefDif,allowLists,actParents,learnerAdditions)]

    with Pool(len(args)) as p:
        worlds = p.starmap(model4do4Func, args)

    makeGraphs(worlds=worlds, labels=[a[0] for a in args], 
               title=name,
               xlabel='time',
               name = name)  
    
if __name__ == '__main__':

    priortype = args.priortype

    dists = [ [0.0, 0.0, 1.0], [0.1, 0.1, 0.8], [0.1, 0.1, 0.8], [0.4, 0.4, 0.2], [0.4, 0.4, 0.2] ]
    
    priors = dists[priortype]

    beliefDif = [0,4,15,4,15][priortype]

    firsthalfacts = np.array(range(NUMACTS)) % 2 == 0
    secondhalfacts = np.array(range(NUMACTS)) % 2 == 1
    allacts = np.ones(NUMACTS, dtype=bool)
    
    allowLists = [{}, {0: firsthalfacts, int(STEPS/3):allacts, int(2*STEPS/3): secondhalfacts}]
    #allowLists = [{}, {0:  firsthalfacts, int(STEPS/3):secondhalfacts, int(2*STEPS/3): firsthalfacts}]

    actParents = [ [-1] * NUMACTS, np.where(np.array(range(NUMACTS))%2==1, range(-1,NUMACTS-1), [-1]*NUMACTS) ]

    learnerAdditions = eval(args.learnerAdditions)


    name = args.outdir + "/"
    #print(allowLists)
    if args.name:
        name += args.name
    else:
        name += args.model + "-p" + str(priortype)+"-t"+str(args.threshold)+"-l"+str(args.allowlists)+"-a"+str(args.actparents) + "-f"+str(int(args.fast)) + "-p" + str(int(args.perfect))
        if args.seed:
            name += "-s" + str(args.seed)
        if args.fast:
            name += "-f"
        if len(learnerAdditions) > 0:
            name += "-la" + "_".join(["%d_%d" % (t,learnerAdditions[t]) for t in learnerAdditions])

    globals()[args.model](priors=priors, beliefDif=beliefDif, name=name,
                          allowLists=allowLists[args.allowlists], actParents=actParents[args.actparents],
                          learnerAdditions=learnerAdditions)


# Try "make people know everything"
# Try "get final total production" (per gender?)
# run the code with prior 0, check it makes no difference
