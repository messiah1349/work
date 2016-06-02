import scipy
from scipy.stats import chisquare
import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import rcParams

def listSum(l1,l2):
    return [x + y for x, y in zip(l1,l2)]

def getExpected(zs,ons):
    zsum = sum(zs)
    osum = sum(ons)
    alls = listSum(ons, zs)
    allsum = sum(alls)
    zsExpected = list(map(lambda x: x * zsum / allsum, zs))
    onExpected = list(map(lambda x: x * osum / allsum, ons))
    expected = zsExpected + onExpected
    return expected

def findSubsets(lst,lLen):
    return list(itertools.combinations(lst, lLen))

def getSubsets(srcl,maxBins):
    subsetsIxs = []
    for subLen in range(1,min(srcl,maxBins)):
        subsetsIxs += list(findSubsets(list(range(1,srcl)),subLen))
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

def bucketSum(lst,ts):                           #из списка lst делает бакетированный список с перегородками из ts
    ms = [sum(lst[0:ts[0]])]
    for i in range(len(ts)-1):
        ms.append(sum(lst[ts[i]:ts[i+1]]))
    ms.append(sum(lst[ts[-1]:]))
    return ms

def checkBucketRate(lst,rateLimit):                        #проверяет на то что все значения списка имеют вес > 5%
    if float(min(lst)) / sum(lst) < rateLimit:
        return False
    else:
        return True

def getBestBuck2(src,zs,ons,rateLimit):                     #выбирает лучшую разбивку с помощью хи2
    chisq = 0
    bestBuck = []
    subsets = getSubsets(len(src))
    for sub in subsets:
        zSubs = bucketSum(zs,sub)
        oSubs = bucketSum(ons,sub)
        if not checkBucketRate(listSum(zSubs,oSubs)):
            continue
        observed_values = scipy.array(zSubs + oSubs)
        expected_values = scipy.array(getExpected(zSubs,oSubs))
        chisqCur = chisquare(observed_values, f_exp=expected_values)[0]
        if chisqCur > chisq:
            chisq = chisqCur
            bestBuck = sub
    return bestBuck

def getBestBuck(src,zs,ons,rateLimit,minBins,maxBins):                     #выбирает лучшую разбивку с помощью хи2
    chisq = 0
    bestBuck = []
    
    while len(bestBuck)==0 and minBins > 1:   #если с данным minBins разбиение невозможно, то уменьшаем minBins
        subsets = getSubsets(len(src),maxBins)
        subsets = [x for x in subsets if (len(x) >= min(len(src)-1,minBins-1)) and (len(x) <= maxBins-1)]
        for sub in subsets:
            zSubs = bucketSum(zs,sub)
            oSubs = bucketSum(ons,sub)
            if not checkBucketRate(listSum(zSubs,oSubs),rateLimit):
                continue
            observed_values = scipy.array(zSubs + oSubs)
            expected_values = scipy.array(getExpected(zSubs,oSubs))
            chisqCur = chisquare(observed_values, f_exp=expected_values)[0]
            if chisqCur > chisq:
                chisq = chisqCur
                bestBuck = sub
        minBins -= 1
    return bestBuck

def getVectFromColumns(df,buckColumn,badColumn):  #возвращает векторы со значениями и соответствующим им количеством
    dfPos = df[df[badColumn]==0.0]                # нулей и единиц 
    dfNeg = df[df[badColumn]==1.0]
    gPos = dfPos.groupby(buckColumn).size().reset_index().rename(columns={0:'posCnt'})
    gNeg = dfNeg.groupby(buckColumn).size().reset_index().rename(columns={0:'negCnt'})
    gAll = pd.merge(gPos,gNeg,how='outer').fillna(0).sort_values(buckColumn)
    gAll[['posCnt','negCnt']] = gAll[['posCnt','negCnt']].astype(float)
    src = gAll[buckColumn].values
    zs = gAll['posCnt'].values
    ons = gAll['negCnt'].values
    return src, zs, ons

def getClustValues(x,src,bb):                     #возвращает номер кластера для для данного значения
    if np.isnan(x):
        return -1
    for num in range(len(bb)):
        if x < src[bb[num]]:
            return num
    return len(bb)

def getClustColumn(df,buckColumn,badColumn,rateLimit,minBins,maxBins):      #из колонки buckColumn создает новую бакетированную
    buffDf = df[[buckColumn,badColumn]].copy()
    newColumn = buckColumn + '_Clust'
    #buffDf[buckColumn] = buffDf[buckColumn].fillna(-1000000)
    if len(buffDf[buffDf[buckColumn].isnull()])==0:
        src, zs, ons = getVectFromColumns(buffDf,buckColumn,badColumn)
        bb = getBestBuck(src,zs,ons,rateLimit,minBins,maxBins)
        buffDf[newColumn] = buffDf[buckColumn].apply(getClustValues,args=[src,bb])
    else:
        dfreal = buffDf[buffDf[buckColumn].notnull()].copy()
        src, zs, ons = getVectFromColumns(dfreal,buckColumn,badColumn)
        bb = getBestBuck(src,zs,ons,rateLimit,minBins,maxBins)
        buffDf[newColumn] = buffDf[buckColumn].apply(getClustValues,args=[src,bb]).fillna(-1)
    return buffDf

def getWOE(x,maxValv,WOEv):                        #для каждого x выбирает соответствующий ему WOE
    if np.isnan(x):
        return WOEv[0]
    for i in range(len(maxValv)):
        if x <= maxValv[i]:
            return WOEv[i]
    return WOEv[-1]

def descrVarTable(df,buckColumn,badColumn,rateLimit,minBins,maxBins):
    buffDf = getClustColumn(df,buckColumn,badColumn,rateLimit,minBins,maxBins) 
    grouped = buffDf.groupby(buckColumn+'_Clust', as_index = False)
    aggDf = pd.DataFrame(grouped.min()[buckColumn])
    aggDf.columns = ['minVal']
    aggDf['maxVal'] = grouped.max()[buckColumn]
    aggDf['bads'] = grouped.sum()[badColumn]
    aggDf['total'] = grouped.size().reset_index()[0].astype(float)
    aggDf['goods'] = aggDf.total - aggDf.bads
    aggDf['badRate'] = aggDf.bads / aggDf.bads.sum()
    aggDf['goodRate'] = aggDf.goods / aggDf.goods.sum()
    aggDf['WOE'] = np.log(aggDf.badRate / aggDf.goodRate) * 100
    return aggDf

def getWOEcolumn(df,buckColumn,badColumn,rateLimit,minBins,maxBins):          # создает новый столбец с соответствующими значениями WOE для buckColumn
    aggDf = descrVarTable(df,buckColumn,badColumn,rateLimit,minBins,maxBins)
    maxValv = aggDf.maxVal.values
    WOEv = aggDf.WOE.values
    df[buckColumn + '_WOE'] = df[buckColumn].apply(getWOE,args=[maxValv,WOEv])
    return df, aggDf

def transformWoeVarsInfo(woeVarsInfo,newPositions):
    newWOE = woeVarsInfo.copy()
    for var in newPositions:
        varPositions = newPositions[var].copy()
        thisVarWOE = woeVarsInfo[woeVarsInfo['variable']==var]
        varPositions.append(len(thisVarWOE))
        newVoeInfo = pd.DataFrame(columns=thisVarWOE.columns)
        minValL, maxValL, badsL, totalL, goodsL = [],[],[],[],[]
        for num in range(len(varPositions)-1):
            curBuccket = thisVarWOE[varPositions[num]:varPositions[num+1]]
            minValL.append( float(pd.DataFrame(curBuccket.min()).loc['minVal',:]) )
            maxValL.append( float(pd.DataFrame(curBuccket.max()).loc['maxVal',:]) )
            badsL.append( float(pd.DataFrame(curBuccket.sum()).loc['bads',:]) )
            totalL.append( float(pd.DataFrame(curBuccket.sum()).loc['total',:]) )
            goodsL.append( float(pd.DataFrame(curBuccket.sum()).loc['goods',:]) )
        newVoeInfo['minVal'] = minValL
        newVoeInfo['maxVal'] = maxValL
        newVoeInfo['bads'] = badsL
        newVoeInfo['total'] = totalL
        newVoeInfo['goods'] = goodsL
        newVoeInfo['badRate'] = newVoeInfo.bads / newVoeInfo.bads.sum()
        newVoeInfo['goodRate'] = newVoeInfo.goods / newVoeInfo.goods.sum()
        newVoeInfo['WOE'] = np.log(newVoeInfo.badRate / newVoeInfo.goodRate) * 100
        newVoeInfo['variable'] = var
        newVoeInfo['rate'] = newVoeInfo.bads / newVoeInfo.total
        newWOE = pd.concat([newWOE[newWOE['variable']!=var],newVoeInfo])
    return newWOE

def twinplot(df,lab):
    a = df.groupby(lab).agg([np.mean, np.size])['badMob3'].reset_index()
    
    minBad = a['mean'].min()*100
    maxBad = a['mean'].max()*100
    
    maxCount = a['size'].max()
    
    f, ax1 = plt.subplots()
    
    xval = list(range(len(a)))
    xCor = [x-0.25 for x in xval]
    ax1.bar(xCor,a['size'],width=0.5, color='g')
    ax1.set_ylabel('Count', color='b')
    ax1.set_ylim(- 0.1*maxCount, 1.1 * maxCount)
    ax1.set_xlabel(lab)
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
        
    for i,j in zip(xval,a[lab]):
        ax1.annotate(str(round(j,1)),xy=(i,-0.06*maxCount))

    ax2 = ax1.twinx()
    ax2.plot(xval,a['mean']*100,'r-')
    ax2.set_ylim(minBad-2,maxBad+1)
    ax2.set_ylabel('BadPercent', color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    for i,j in zip(xval,a['mean']*100):
        ax2.annotate(str(round(j,1)),xy=(i-0.1,j+0.3))

    plt.show()
    
def twinPlotWoe(df):
    dfWOE = df.copy()
    variables = dfWOE.variable.unique()
    dfWOE['rate'] = dfWOE.bads / dfWOE.total
    dfWOE['minVal'] = dfWOE['minVal'].fillna(-1000000)
    for var in variables:
        dfVar = dfWOE[dfWOE.variable == var]
        minBad = dfVar.rate.min()*100
        maxBad = dfVar.rate.max()*100
        maxCount = dfVar.total.max()
        
        f, ax1 = plt.subplots()
    
        xval = list(range(len(dfVar)))
        xCor = [x-0.25 for x in xval]
        ax1.bar(xCor,dfVar['total'],width=0.5, color='g')
        ax1.set_ylabel('Count', color='b')
        ax1.set_ylim(- 0.1*maxCount, 1.1 * maxCount)
        ax1.set_xlabel(var)
        for tl in ax1.get_yticklabels():
            tl.set_color('b')

        for i,j in zip(xval,dfVar.minVal):
            ax1.annotate(str(round(j,1)),xy=(i,-0.06*maxCount))

        ax2 = ax1.twinx()
        ax2.plot(xval,dfVar['rate']*100,'r-')
        ax2.set_ylim(minBad-2,maxBad+1)
        ax2.set_ylabel('BadPercent', color='r')
        for tl in ax2.get_yticklabels():
            tl.set_color('r')
        for i,j in zip(xval,dfVar['rate']*100):
            ax2.annotate(str(round(j,1)),xy=(i-0.1,j+0.3))