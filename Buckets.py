import scipy
from scipy.stats import chisquare
import itertools

def getExpected(zs,ons):
    zsum = sum(zs)
    osum = sum(ons)
    alls = [x + y for x, y in zip(src, zs)]
    allsum = sum(alls)
    zsExpected = list(map(lambda x: x * zsum / allsum, zs))
    onExpected = list(map(lambda x: x * osum / allsum, ons))
    expected = zsExpected + onExpected
    return expected

def getSubsets(srcl):
    subsetsIxs = []
    for subLen in range(1,min(srcl,5)):
        subsetsIxs += list(findsubsets(list(range(1,srcl)),subLen))
    return subsetsIxs

def bucketsName(src,ts):  
    if ts[0]==1:
        bk1 = str(src[0])
    else:
        bk1 = str(src[0]) + '-' + str(src[ts[0]-1])
    bk = [bk1]

    for i in range(len(ts)-1):
        if ts[i+1] - ts[i] == 1:
            bk.append(str(src[ts[i]]))
        else:
            bk.append(str(src[ts[i]]) + '-' + str(src[ts[i+1]-1]))
    if ts[-1]==srcl-1:
        bk.append(str(src[ts[-1]]))
    else:
        bk.append(str(src[ts[-1]]) + '-' + str(src[-1]))
    return bk

def bucketSum(lst,ts):
    ms = [sum(lst[0:ts[0]])]
    for i in range(len(ts)-1):
        ms.append(sum(lst[ts[i]:ts[i+1]]))
    ms.append(sum(lst[ts[-1]:]))
    return ms

def getBestBuck(src,zs,ons): 
    chisq = 0
    bestBuck = []
    subsets = getSubsets(len(src))
    for sub in subsets:
        zSubs = bucketSum(zs,sub)
        oSubs = bucketSum(ons,sub)
        observed_values = scipy.array(zSubs + oSubs)
        expected_values = scipy.array(getExpected(zSubs,oSubs))
        chisqCur = chisquare(observed_values, f_exp=expected_values)[0]
        if chisqCur > chisq:
            chisq = chisqCur
            bestBuck = sub
    return bestBuck