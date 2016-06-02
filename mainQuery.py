query = """
select
	
	c.clientKey
	,c.guid
	,la.ApplicationDate
	,t.name as traderName
	,t.traderKey
    
    ,fa.[ApplicationId]
    ,fa.[CreditScore]
    ,fa.[CardId]
    ,fa.[FoStore]
    ,fa.[PosPolicyGroup]
    ,fa.[RegionStore]
    ,fa.[Age]
    ,fa.[CityStore]
    ,fa.[GenderTypeKey]
    ,fa.[InitialLimit]
    ,fa.[RetailerChain]
    ,fa.[RetailerPos]
    ,fa.[BkiFlg]
    ,fa.[PosBureauGroup]
    ,fa.[Segment]
    ,fa.[FinalDecision]
    ,fa.[PostBureau]
    ,fa.[PreBureau]
    ,fa.[FirstDeclineRule]
    ,fa.[Repeated]
    ,fa.[Limit]
    ,fa.[Source]
    ,fa.[Strategy]
    ,fa.[RiskGrade]
    ,fa.[ProbDef]
    ,fa.[ProbResp]
    ,fa.[MaxGapDays]
    ,fa.[MaxDifferenceInDays]
    ,fa.[TotalMaxPayment]
    ,fa.[TotalPaymentsCount]
    ,fa.[TotalPaymentsSum]
    ,fa.[LifePeriod]

	,fcr.[ApplicationId]
    ,fcr.[FirstLoanDate]
    ,fcr.[Inquiry12Month]
    ,fcr.[Inquiry1Month]
    ,fcr.[Inquiry1Week]
    ,fcr.[Inquiry3Month]
    ,fcr.[Inquiry6Month]
    ,fcr.[Inquiry9Month]
    ,fcr.[InquiryRecentPeriod]
    ,fcr.[LastLoanDate]
    ,fcr.[LoansActive]
    ,fcr.[LoansActiveMainBorrower]
    ,fcr.[LoansMainBorrower]
    ,fcr.[MaxOverdueStatus]
    ,fcr.[PayLoad]
    ,fcr.[TtlAccounts]
    ,fcr.[TtlBankruptcies]
    ,fcr.[TtlConsumer]
    ,fcr.[TtlCreditCard]
    ,fcr.[TtlDelq3059]
    ,fcr.[TtlDelq3059L12m]
    ,fcr.[TtlDelq30L12m]
    ,fcr.[TtlDelq5]
    ,fcr.[TtlDelq529]
    ,fcr.[TtlDelq6089]
    ,fcr.[TtlDelq6089L12m]
    ,fcr.[TtlDelq90Plus]
    ,fcr.[TtlDelq90PlusL12m]
    ,fcr.[TtlInquiries]
    ,fcr.[TtlLegals]
    ,fcr.[TtlOfficials]
    ,fcr.[WorstStatusEver]

from
	client c
		join
	(
		select
			*
			,Row_number() over (partition by clientKey order by ficoapplicationKey) as Farn
		from
			ficoapplication 
	)fa
on
	c.clientKey = fa.clientKey
		join
	loanApplication la
on
	la.loanApplicationKey = fa.loanApplicationKey
		join
	store s
on
	la.storeKey = s.storeKey
		join
	trader t
on
	t.traderKey = s.traderKey
		join
	(
		select
			*
			,Row_number() over (partition by ficoapplicationKey order by FicoCreditRegistryKey) as fcrRn
		from
			FicoCreditRegistry 
	)fcr
on
	fcr.ficoApplicationKey = fa.ficoApplicationKey
where
	faRn = 1 and fcrRn = 1
	"""