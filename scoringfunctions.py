import pandas as pd
import numpy as np
import pandas.core.algorithms as algos
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics as met
from sklearn.ensemble import ExtraTreesClassifier
from pylab import rcParams
from sklearn.utils import check_consistent_length, column_or_1d, check_array
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import seaborn as sns
import woe
import ipywidgets as widgets

def colCnt(df):
    d = []
    for col in df.columns:
        d.append((col,len(df[col].unique())))  

    colCntDescr = pd.DataFrame(d,columns=['col','uinqValuesCount'])
    return colCntDescr
    
def getCategoricalColumn(df,column,clustNum):
    df = df.reset_index(drop=True)
    
    if df[column].min() == 0:                                     #если есть 0, то его берем в отдельный кластер
        dfColNotZero = df.loc[df[column]>0,column]
        bins = pd.core.algorithms.quantile(np.unique(dfColNotZero),np.linspace(0, 1, clustNum))
        bins = np.insert(bins,0,0)
    else:
        bins = pd.core.algorithms.quantile(np.unique(df[column]),np.linspace(0, 1, clustNum+1))
    
    result = pd.tools.tile._bins_to_cuts(df[column], bins, include_lowest=True)   # колонка в виде категорий
    categoricalTable = pd.DataFrame(list(zip(df[column],result)),columns=[column,'categorical']) # сцепка исходной колонки с категориями
    newColumnName = column + '_cat'     #имя новой колонки
    grouped = categoricalTable.groupby('categorical')
    minCatValue = grouped.min().reset_index().rename(columns={column:newColumnName}) 
    categoricalTable = pd.merge(categoricalTable,minCatValue,how='left') #добавляем минимальное значение в категории, это будет имя нашего кластера
    dfr = df.copy()
    dfr[newColumnName] = categoricalTable[newColumnName]  #исходный датафрейм с новой колонкой 
    minCatValue['maxVal'] = grouped.max()[column].values
    minCatValue.rename(columns = {newColumnName: 'minVal'},inplace=True)
    return dfr, minCatValue

def getClustColumnTransform(x,minVal,maxVal):
    for mn, mx in zip(minVal,maxVal):
        if x <= mx:
            return mn
    return minVal[-1]
        
def getClustTransform(df,clustVarsInfo):
    dfOut = df.copy()
    clustVars = clustVarsInfo.variable.unique()
    for var in clustVars:
        varTransformTable = clustVarsInfo[clustVarsInfo.variable==var].sort_values('maxVal')
        newColumnName = var + '_cat'
        dfOut[newColumnName]=dfOut[var].apply(getClustColumnTransform,args=[list(varTransformTable.minVal),list(varTransformTable.maxVal)])
    dfOut = dfOut.drop(clustVars,axis=1)
    return dfOut

def getWOETransform(df,woeVarsInfo):
    dfOut = df.copy()
    woeVars = woeVarsInfo.variable.unique()
    for var in woeVars:
        varTransformTable = woeVarsInfo[woeVarsInfo.variable==var]
        newColumnName = var + '_WOE'
        dfOut[newColumnName]=dfOut[var].apply(getWOEColumnTransform,args=[list(varTransformTable.maxVal),list(varTransformTable.WOE)])
    dfOut = dfOut.drop(woeVars,axis=1)
    return dfOut

def getWOEColumnTransform(x,maxVal,WOE):
    if np.isnan(x) & np.isnan(maxVal[0]):
        return WOE[0]
    for mx, w in zip(maxVal,WOE):
        if x <= mx:
            return w
    return WOE[-1]

def continuousVariables(df,columnLimit=50):
    colCntDescr = colCnt(df)
    
    clustVarsInfo = pd.DataFrame(columns = ['categorical','minVal','maxVal','variable'])

    dfPreWoe = df.copy()

    continuousVars = colCntDescr.loc[colCntDescr['uinqValuesCount']>columnLimit,'col'].values

    for col in continuousVars:
        dfPreWoe, clusColInfo = getCategoricalColumn(dfPreWoe,col,columnLimit)
        clusColInfo['variable'] = col
        clustVarsInfo = pd.concat([clustVarsInfo,clusColInfo])

    dfPreWoe = dfPreWoe.drop(continuousVars,axis = 1 )
    
    return dfPreWoe, clustVarsInfo

def woeVariables(df,badFlag,rateLimit=0.05,minBins=2,maxBins=5,columnChoose=False,columns=[]):
    if columnChoose:
        colCntDescrPreWoe = colCnt(df[columns])
    else:
        colCntDescrPreWoe = colCnt(df)
    colCntDescrPreWoe = colCntDescrPreWoe[colCntDescrPreWoe['col']!=badFlag]
    woeColumns = list(colCntDescrPreWoe.loc[colCntDescrPreWoe['uinqValuesCount']>1,'col'])
    
    woeVarsInfo = pd.DataFrame(columns = ['minVal', 'maxVal', 'bads', 'total', 'goods', 'badRate', 'goodRate','WOE','variable'])

    dfPostWoe = df.copy()
    
    colPos = 0.0
    totalCols = len(woeColumns)
    print('Progress: ',end="")
    for woeColumn in woeColumns:
        dfPostWoe, woeColInfo = woe.getWOEcolumn(dfPostWoe,woeColumn,badFlag,rateLimit,minBins,maxBins) 
        woeColInfo['variable'] = woeColumn
        woeVarsInfo = pd.concat([woeVarsInfo,woeColInfo])
        colPos+=1.0
        print("%.1f%%, " % (colPos/totalCols * 100),end="")
    dfPostWoe = dfPostWoe.drop(woeColumns,axis=1)
    print("")
    return dfPostWoe, woeVarsInfo

def getWOEcolumnAfterTransform(dfS,woeVarsInfo):
    df = dfS.copy()
    WOEvariables = list(woeVarsInfo.variable.unique())
    for var in WOEvariables:
        #print(var)
        WOEvarsInfoCur = woeVarsInfo[woeVarsInfo.variable==var]
        maxValv = WOEvarsInfoCur.maxVal.values
        WOEv = WOEvarsInfoCur.WOE.values
        newColumn = var + '_WOE'
        df[newColumn] = df[var].apply(woe.getWOE,args=[maxValv,WOEv])
    df = df.drop(WOEvariables,axis=1)
    return df

def corrTable(df,informationTable):
    CorrKoef = df.corr()
    corColumns = CorrKoef.columns
    c1 = []
    c2 = []
    corVal = []
    for i in range(1,len(CorrKoef)):
        for j in range(i+1,len(CorrKoef)):
            if CorrKoef.iloc[i,j]>0.6:
                c1.append(corColumns[i])
                c2.append(corColumns[j])
                corVal.append(CorrKoef.iloc[i,j])
    corDf = pd.DataFrame({'var1':c1,'var2':c2,'r^2':corVal}).reindex_axis(['var1','var2','r^2'],axis=1).sort_values('r^2',ascending=False)
    allCorVars = list(corDf.var1) + list(corDf.var2)
    dfO = pd.DataFrame({'variable':allCorVars})
    dfOg = dfO.groupby('variable').size().reset_index().sort_values(0,ascending=False)
    dfOg = pd.merge(dfOg,informationTable[['variable','informationValue']])
    dfOg = dfOg.rename(columns = {0:'varCorrelationCount'})
    return corDf,dfOg

def getIVfromWOE(woeDf):
    ivs = []
    variables = woeDf.variable.unique()
    for var in variables:
        vardf = woeDf[woeDf.variable==var]
        iv = -((vardf.goodRate - vardf.badRate) * vardf.WOE).sum() / 100
        ivs.append(iv)
    dfOut = pd.DataFrame(list(zip(variables,ivs)),columns = ['variable','InformationValue'])
    dfOut = dfOut.sort_values('InformationValue',ascending=False)
    return dfOut

def preClean(x):
    if x[-4:]!='_WOE': a = x
    else: a = x[:-4]
        
    if a[-4:]!='_cat': return a
    else: return a[:-4]
   

def columnClean(columns):
    cwoCAT = list(map(preClean,columns))
    return cwoCAT

def ootTransform(ootDf,clustVarsInfo,woeVarsInfo,goodColumns):
    
    dfOotPreWOE = getClustTransform(ootDf,clustVarsInfo)
    dfOotPostWOE = getWOETransform(dfOotPreWOE,woeVarsInfo)
 
    columnCleans = columnClean(goodColumns)
    d = dict(zip(list(dfOotPostWOE.columns),columnClean(dfOotPostWOE.columns)))
    outpColumns = [k for k,v in d.items() if v in columnCleans]
     
    dfFPDpreLR = dfOotPostWOE[outpColumns]
    return dfFPDpreLR

def featureImportance(X,y,columns):
    model = ExtraTreesClassifier()
    model.fit(X,y)
    fi = pd.DataFrame(np.array([columns,model.feature_importances_]).T,columns=['variable','feature_importance']).sort_values('feature_importance',ascending = False)
    return fi

def giniGrowth(df,woeVarsInfo,badFlag):
    woeTable = woeVarsInfo.copy()
    woeTable.variable = woeTable.variable.apply(lambda x: x + '_WOE')
    IV = getIVfromWOE(woeTable)
    columns = IV.variable
    columnsForModeking = []
    giniTest = []
    giniTrain = []
    y = df[badFlag].values
    for col in columns:
        columnsForModeking.append(col)
        X = df[columnsForModeking].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=3)
        lr = LogisticRegression()
        lr.fit(X_train,y_train)
        pr_test = lr.predict_proba(X_test)[:,1]
        pr_train = lr.predict_proba(X_train)[:,1]
        rocGiniTest =  met.roc_auc_score(y_test,pr_test) * 2 - 1
        rocGiniTrain =  met.roc_auc_score(y_train,pr_train) * 2 - 1
        giniTest.append(rocGiniTest)
        giniTrain.append(rocGiniTrain)
    trainDiff = [x-y for x,y in zip(giniTrain,[0]+giniTrain[:-1])]
    testDiff = [x-y for x,y in zip(giniTest,[0]+giniTest[:-1])]
    dfOut = pd.DataFrame({'variable':columns, 'giniTrain' : giniTrain,'giniTest': giniTest,'trainDiff':trainDiff,'testDiff':testDiff,'informationValue':list(IV.InformationValue)})
    dfOut[['trainDiff','testDiff']] = dfOut[['trainDiff','testDiff']]#.apply('${:,.2f}'.format)
    dfOut = dfOut.reindex_axis(['variable','informationValue','testDiff','trainDiff','giniTest','giniTrain'],axis=1)
    return dfOut

def woeOutput(woeInfoTrans,goodColumns,coef_,factor):
    woeInfoOutput = woeInfoTrans.copy()
    goodC = [x[:-4] for x in goodColumns]
    d = dict(zip(goodC,coef_.tolist()[0]))
    woeInfoOutput = woeInfoOutput[woeInfoOutput['variable'].isin(list(d.keys()))]
    woeInfoOutput['coef'] = ''
    for var in d:
        woeInfoOutput.loc[woeInfoOutput['variable']==var,'coef']=d[var]
    woeInfoOutput['scorValue'] = -woeInfoOutput['WOE']*woeInfoOutput['coef']*factor    
    woeInfoOutput = woeInfoOutput[['variable','minVal','maxVal','scorValue']]
    return woeInfoOutput

def woeProduction(woeOutp):
    woeOut = woeOutp.copy()
    woeOut['varInit'] = woeOut.variable.apply(lambda x: x if x[-4:]!='_cat' else x[:-4] )
    variables = list(woeOut.varInit.unique())
    for var in variables:
        woeOut.loc[(woeOut.varInit==var)&(woeOut.maxVal.notnull()),
                   'maxVal'] = list(woeOut.loc[(woeOut.varInit==var)&(woeOut.maxVal.notnull()),'minVal'])[1:] + [100000000]
    woeOut = woeOut[['varInit','maxVal','scorValue']]
    return woeOut

def getScoringColumn(srcRow,woeOutp,intercept):
    woeOut=woeOutp.copy()
    woeOut['varInit'] = woeOut.variable.apply(lambda x: x if x[-4:]!='_cat' else x[:-4])
    variables = list(woeOut.varInit.unique())
    ans = intercept
    for var in variables:
        #print(var)
        woeVar = woeOut[woeOut.varInit==var]
        if np.isnan(srcRow[var]):
            ans += woeVar.loc[woeVar.minVal.isnull(),'scorValue'].values[0]
        else:
            woeVar = woeVar[woeVar.minVal.notnull()]
            maxVals = zip(list(woeVar.minVal)[1:]+[1000000],list(woeVar.scorValue))
            for i,j in maxVals:
                if srcRow[var] < i:
                    ans+=j
                    #print('srcRow[var] = %s, scorVal = %s' %(srcRow[var],j))
                    break
            else: 
                ans+=list(woeVar['scorValue'])[-1]
                #print(list(woeVar['scorValue'])[-1])
    return ans

def variableToScoringValue(x,mv,sv):
    ######mv = wouOutpVar.minVal.values
    #sv = wouOutpVar.scorValue.values
    if np.isnan(mv[0]):
        svT = sv[1:]
        mvT = np.concatenate([mv[2:],[1000000]])
    else:
        svT = sv
        mvT = np.concatenate([mv[1:],[1000000]])
    if np.isnan(x):
        return sv[0]
    else:
        l = len(np.where(mvT<=x)[0])
        return svT[l]

def getScoringTable(dfSrc,woeOut,intercept,IdColumn):
    df = dfSrc.copy()
    woeOut['varInit'] = woeOut.variable.apply(preClean)
    variables = list(woeOut['varInit'].unique())
    weightsDf = pd.DataFrame(np.zeros((len(df),len(variables))),columns = variables)
    
    for var in variables:
        woeVar = woeOut[woeOut.varInit==var]
        mv = woeVar.minVal.values
        sv = woeVar.scorValue.values
        weightsDf[var] = df[var].apply(variableToScoringValue,args=[mv,sv]).values
    
    weightsDf['scoring'] = weightsDf.sum(axis = 1) + intercept 
    weightsDf[IdColumn] = df[IdColumn].values
    weightsDf = weightsDf.reindex_axis([IdColumn,'scoring']+variables, axis=1)
    
    return weightsDf


def bucketRate(y_true,y_score,buckets=10):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    ysz = y_true.size
    ysm = y_true.sum()
    desc_score_indices = np.argsort(y_score, kind="mergesort") #[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    ixsSec = [int(ysz*(i/buckets))-1 for i in range(1,buckets+1)]
    ixsFirst = [0] + ixsSec[:-1]
    cummBadRate = []
    curBadRate = []
    cutOff = []
    for i in range(buckets):
        cummBadRate.append(y_true[:ixsSec[i]].mean())
        curBadRate.append(y_true[ixsFirst[i]:ixsSec[i]].mean())
        cutOff.append(y_score[ixsSec[i]])
    
    arBorders = np.arange(0, 1, 1/buckets) + 1/buckets
    arBorders = list(map(lambda x: round(x,0), list(arBorders*100)) )
    arBordersCur = list(map(lambda x,y: str(int(x)) + "-" + str(int(y)), [0]+arBorders[:-1], arBorders))
    cummBadRate = list(map(lambda x: round(x*100,1) ,cummBadRate))
    curBadRate = list(map(lambda x: round(x*100,1) ,curBadRate))
    
    df = pd.DataFrame({'AR_cum':arBorders, 'cumBadRate':cummBadRate,
                       'AR_cur': arBordersCur, 'curBadRate':curBadRate,
                      'CutOff': cutOff}).sort_values('AR_cum'
                    , ascending=False).reindex_axis(['AR_cum','cumBadRate','AR_cur','curBadRate','CutOff'], axis=1)
    return  df

def decAR(x,brr):
    l = len(brr)
    for i in range(1,l+1):
        if x <= brr[l-i]:
            return float(i) * 100 / l

def rocAuc(y_true,y_score):
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    fps, tps, _ = met.roc_curve(y_true,y_score, pos_label=1)
    rocGini =  met.roc_auc_score(y_true,y_score) * 2 - 1
    return fps, tps, rocGini

def rocCurve(y_true,y_score):
    
    fps, tps, rocGini = rocAuc(y_true,y_score)

    plt.figure()
    plt.plot(fps, tps, label='ROC curve (rocGini = %.4f)' % (rocGini))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
def сheckIndex(s):
    if s.index.dtype_str=='category':
        return True
    else: return False

def warm2Columns(dfSrc,column1,column2,badFlag,binSize=10,countLimit=200):
    df = dfSrc[[column1,column2,badFlag]].copy()
    bins1 = np.unique(algos.quantile(df[column1], np.linspace(0, 1, binSize+1)))
    bins2 = np.unique(algos.quantile(df[column2], np.linspace(0, 1, binSize+1)))
    df[column1+'_bin'] = pd.tools.tile._bins_to_cuts(df[column1], bins1, include_lowest=True)
    df[column2+'_bin'] = pd.tools.tile._bins_to_cuts(df[column2], bins2, include_lowest=True)
    pvMean = df.pivot_table(badFlag,column1+'_bin',column2+'_bin',np.mean).fillna(0)
    pvSize = df.pivot_table(badFlag,column1+'_bin',column2+'_bin',np.size).fillna(0)
    
    if сheckIndex(pvSize):
        for ind in pvSize.index:
            for col in pvSize.columns:
                if np.isnan(pvSize.loc[ind,col].values[0][0]):
                    pvMean.loc[ind,col]=0
                elif pvSize.loc[ind,col].values[0][0]<countLimit:
                    pvMean.loc[ind,col]=0
    else:
        for ind in pvSize.index:
            for col in pvSize.columns:
                if np.isnan(pvSize.loc[ind,col]):
                    pvMean.loc[ind,col]=0
                elif pvSize.loc[ind,col] < countLimit:
                    pvMean.loc[ind,col]=0

    ss = sns.heatmap(pvMean,annot=True)
    return pvMean,pvSize


