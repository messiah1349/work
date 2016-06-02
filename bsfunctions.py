import pandas as pd
import datetime

def MaxOverdueStatusClust(x):
    if x == 'A':
        return 0.0
    x = float(x)
    if x==0 :
        return 1.0
    elif x == 1:
        return -1.0
    else:
        return x
    
def WorstStatusEverClust(x):
    if x == 'A':
        return -1.0
    elif x== 'X':
        return 0.0
    x = float(x)
    if x==0.0 :
        return 1.0
    elif x==1.0:
        return -2.0
    else:
        return x
    
def mainTransform(df):
    dfVar = df[['FirstDeclineRule','ApplicationDate','Age','GenderTypeKey','BkiFlg','FirstLoanDate','Inquiry12Month', 'Inquiry1Month',
       'Inquiry1Week', 'Inquiry3Month', 'Inquiry6Month', 'Inquiry9Month',
       'InquiryRecentPeriod', 'LastLoanDate', 'LoansActive',
       'LoansActiveMainBorrower', 'LoansMainBorrower', 'MaxOverdueStatus',
       'PayLoad', 'TtlAccounts', 'TtlConsumer',
       'TtlCreditCard', 'TtlDelq3059', 'TtlDelq3059L12m', 'TtlDelq30L12m',
       'TtlDelq5', 'TtlDelq529', 'TtlDelq6089', 
       'TtlDelq90Plus',  'TtlInquiries', 
        'WorstStatusEver','traderKey','badMob3','badFpd','clientKey']]

    dfVar = dfVar[dfVar.BkiFlg==1]
    dfVar = dfVar.drop(['BkiFlg'],axis=1)

    dfVar['ApplicationDate'] = pd.to_datetime( dfVar['ApplicationDate'])
    dfVar.FirstLoanDate = pd.to_datetime(dfVar.FirstLoanDate)
    dfVar.LastLoanDate = pd.to_datetime(dfVar.LastLoanDate)

    dfVar['floan'] = (dfVar.ApplicationDate - dfVar.FirstLoanDate).dt.days
    dfVar['lloan'] = (dfVar.ApplicationDate - dfVar.LastLoanDate).dt.days
    dfVar.loc[dfVar.floan<0,'floan'] = 0
    dfVar.loc[dfVar.lloan<0,'lloan'] = 0

    dfVar.MaxOverdueStatus = dfVar.MaxOverdueStatus.apply(MaxOverdueStatusClust)
    dfVar.WorstStatusEver = dfVar.WorstStatusEver.apply(WorstStatusEverClust)

    dfVar['TtlDelq30L12m'] = dfVar['TtlDelq30L12m'].fillna(0)
    dfVar['LoansMainBorrower'] = dfVar['LoansMainBorrower'].fillna(0)
    dfVar['TtlAccounts'] = dfVar['TtlAccounts'].fillna(0)
    dfVar['TtlDelq3059'] = dfVar['TtlDelq3059'].fillna(0)
    dfVar['TtlDelq3059'] = dfVar['TtlDelq3059'].fillna(0)

    dfVar[['TtlDelq30L12m','LoansMainBorrower','TtlAccounts','TtlDelq3059','TtlDelq3059','TtlDelq3059L12m','TtlDelq5',
          'TtlDelq529','TtlDelq6089','TtlDelq90Plus','TtlDelq3059L12m',
           'TtlInquiries','PayLoad','TtlDelq3059L12m']] = dfVar[['TtlDelq30L12m','LoansMainBorrower','TtlAccounts','TtlDelq3059','TtlDelq3059','TtlDelq3059L12m','TtlDelq5',
          'TtlDelq529','TtlDelq6089','TtlDelq90Plus','TtlDelq3059L12m',
           'TtlInquiries','PayLoad','TtlDelq3059L12m']].fillna(0)
    return dfVar

def chooseTrader(df,traderList,lastDate,dateBegin = datetime.datetime(2015,5,1)):
    
    dfTrader = df[(df['ApplicationDate']>dateBegin)&
                          (df.floan.notnull())&
                          (df.traderKey.isin(traderList))]
    
    dfDecline = dfTrader[(dfTrader.ApplicationDate < lastDate)&(dfTrader.badMob3.isnull())&(df.badFpd.isnull())&(dfTrader.FirstDeclineRule.notnull())]
    dfDecline.drop(['ApplicationDate','FirstLoanDate','LastLoanDate','traderKey','badMob3','badFpd'],axis = 1,inplace=True)
    
    dfTrader = dfTrader.drop(['ApplicationDate','FirstLoanDate','LastLoanDate','traderKey','FirstDeclineRule'],axis = 1)
    
    dfMob3 = dfTrader[dfTrader['badMob3'].notnull()].drop('badFpd',axis = 1).reset_index(drop=True)
    dfFPD = dfTrader[dfTrader['badMob3'].notnull()].drop('badMob3',axis = 1).reset_index(drop=True)
    dfOot = dfTrader[(dfTrader['badMob3'].isnull())&(dfTrader['badFpd'].notnull())].drop('badMob3',axis = 1).reset_index(drop=True)
    
    return dfMob3, dfFPD, dfOot, dfDecline