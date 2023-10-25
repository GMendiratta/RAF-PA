### DO NOT MODIFY this file without copying exact modifications into respective functions in fitdata.py
# import libraries
import numpy as np
import scipy.optimize as so
helpnotes="List of Model Functions: \n actkinr() actkin() d2DTOT() DTOT2d() DTOT2AK() DTOT2AKnorm()\n List of Other Functions:\n gendat() checkpos()  solrange() parsepar()\n Some other functions accept a variable fitflag with value of 0,1 or2.\n fitflag=0 allows all parameters to vary.\n fitflag=1 sets Kdim to a value of 0.1\n fitflag=2 sets Kdim=0.1 and RAF=0.04"
# global boundary values
bdyglobal={'f':[10**-5,100],'g':[1,10**4],'KA':[0.1,10**2],'Kd':[10**-4,10**4],'Kdim':[10**-4,10**4],'RAF':[10**-4,10**3]} # Units: micro-Molar
params0={'f':0.01,'g':10.,'KA':2.5,'rafr':0.4,'Kd':0.1,'Kdim':0.1,'RAF':0.04} # dimensionful quantities in units of micro-molar. Sample parameter points. 
DTOTmin=10**-6 # in micro-molar so = 1 pM
def checkpos(x):
    """Returns 'True' if all elements in a list are positive numbers greater than 10^-7 else False"""
    flag=True
    if type(x) == dict:
        x=list(x.values())
    for elem in x:
        if elem<=10**-7:
            flag=False
    return flag
        

def actkinr(dr,params):
    """define the raf activity as a function of normalized parameters and unbound relative drug concentration"""
    try:
        if dr>=0. and checkpos(list(params.values())) and dr!=None:
            return (params['f']*params['g']**2*(params['f'] + dr)*(1 + params['KA'] + dr - np.sqrt((1 + params['KA'] + dr)**2 + (8*params['rafr']*(params['f']*params['g'] + 2*params['g']*dr + dr**2))/(params['f']*params['g'])))**2)/(8*params['rafr']*(params['f']*params['g'] + 2*params['g']*dr + dr**2)**2)
    except:
        print("ERROR actkinr:",dr,params)
        
def actkin(d,params):
    """define the raf activity as a function of un-normalized parameters and unbound drug concentration"""
    try:
        if d>=0. and checkpos(list(params.values())):
            return (params['f']*params['g']**2*params['Kd']*(d + params['f']*params['Kd'])*params['Kdim']*(d + params['Kd'] + params['KA']*params['Kd'] - np.sqrt((d + params['Kd'] + params['KA']*params['Kd'])**2 + (8*(d**2 + 2*d*params['g']*params['Kd'] + params['f']*params['g']*params['Kd']**2)*params['RAF'])/(params['f']*params['g']*params['Kdim'])))**2)/(8*(d**2 + 2*d*params['g']*params['Kd'] + params['f']*params['g']*params['Kd']**2)**2*params['RAF'])
    except:
        print("ERROR actkin:",d,params)
        
def d2DTOT(d,params):
    """Inputs unbound drug concentration alongwith a dictionary of parameters to return the total drug concentration"""
#     if d>=0. and checkpos(list(params.values())):
    try:
        return (d*(8*(d**2 + 2*d*params['g']*params['Kd'] + params['f']*params['g']*params['Kd']**2)**2 - 2*params['f']*params['g']*(d**2 + 2*d*params['g']*params['Kd'] + params['f']*params['g']*params['Kd']**2)*params['Kdim']*(d + params['Kd'] + params['KA']*params['Kd'] - np.sqrt((d + params['Kd'] + params['KA']*params['Kd'])**2 + (8*(d**2 + 2*d*params['g']*params['Kd'] + params['f']*params['g']*params['Kd']**2)*params['RAF'])/(params['f']*params['g']*params['Kdim']))) + d*params['f']*params['g']*params['Kdim']*(d + params['Kd'] + params['KA']*params['Kd'] - np.sqrt((d + params['Kd'] + params['KA']*params['Kd'])**2 + (8*(d**2 + 2*d*params['g']*params['Kd'] + params['f']*params['g']*params['Kd']**2)*params['RAF'])/(params['f']*params['g']*params['Kdim'])))**2 + params['f']*params['g']**2*params['Kd']*params['Kdim']*(d + params['Kd'] + params['KA']*params['Kd'] - np.sqrt((d + params['Kd'] + params['KA']*params['Kd'])**2 + (8*(d**2 + 2*d*params['g']*params['Kd'] + params['f']*params['g']*params['Kd']**2)*params['RAF'])/(params['f']*params['g']*params['Kdim'])))**2))/(8*(d**2 + 2*d*params['g']*params['Kd'] + params['f']*params['g']*params['Kd']**2)**2)
    except:
        print("ERROR: d2DTOT: ",d,params)
        
def DTOT2d(DTOT,params):
    """Numerically solves the inverse function d2DTOT to convert input total drug concentration and parameters into unbound drug conccentration"""
    try:
        DTOT=int(DTOT) if DTOT>10**8 else DTOT # for very large numbers, the floating point precision is irrelevant to precision but crashes optimization algorithms.
        if DTOT>DTOTmin and checkpos(list(params.values())):
            objfn=lambda d:(DTOT-d2DTOT(d,params))/DTOT
            return so.brentq(objfn,10.**-14,DTOT)
        else:
            return 0.
    except:
        print("ERROR DTOT2d:",DTOT,params)
        return 0. #unlikely to provide best fit but doesn't break searching algorithm.

def DTOT2AK(DTOT,params):
#     print("DTOT,params",DTOT,params)
    if DTOT == None:
        d=0.
        DTOT=0.
    try:
        if checkpos(list(params.values())) == False:
            return 10**10
        elif DTOT>=DTOTmin:
            d=DTOT2d(DTOT,params)
        else:
            d=0.
        return actkin(d,params)
    except:
        print("ERROR DTOT2AK:",DTOT,params)
        
def DTOT2AKnorm(DTOT,params):
    """This function inputs total drug values (in uM or same units as Kd in params), paramteres to output active RAF protomers normalized to no-drug"""
    return DTOT2AK(DTOT,params)/actkin(0,params)

def solrange(params):
    """This function inputs a set of absolute parameters and finds the solution for total drug concentration corresponding to maxima, maximal fold change and total drug concentration at which the drug becomes an inhibitor (activity levels equal drug free levels). Drug concentrations are given in micro molar."""
    mindrbound=10**-10 # lower bound for unbound drug to Kd ratio. We ignore inhibitors that show PA at concentrations lower than 1 pM. So, with weakest Kd we explore, 10**4, the unbound drug concentration to Kd ratio cannot be smaller than 10**-10 uM.
    maxdrbound=10**7 # upper bound for unbound drug to Kd ratio. calculated from highest drug dose at 1000uM divided by lowest Kd explored :10**-4 uM
    params['rafr']=params['RAF']/params['Kdim']
    kinref=actkinr(0,params)
    objfn=lambda dr:kinref/actkinr(dr,params)
    drbounds=(mindrbound,maxdrbound)
    try:
        res=so.minimize_scalar(objfn,bounds=drbounds)
        if (res.x<1) or (res.success == False):
            if objfn(mindrbound)>1.0:
            # if no PA exists, no point in proceeding to re-scan. The drug is a pure inhibitor
                resx=0
                drroot=resx
            else:
                res=so.minimize_scalar(objfn,bounds=(mindrbound,1.),method='Bounded') # handles the case where dr values are very small.
        drroot=res.x
    except:# This exception handles cases that are essentially nearly pure-inhibitors (hence second minimization function also fails)
        resx=0
        drroot=resx
    kinmax=actkinr(drroot,params)
    foldchange=kinmax/kinref
    if foldchange>1.:
        Droot=d2DTOT(drroot*params['Kd'],params)
        if Droot>DTOTmin:
            AKref=DTOT2AK(0.,params)
            objfn1=lambda Dtot:(DTOT2AK(Dtot,params)-AKref)/AKref
            try:
                width=so.brentq(objfn1,Droot,10**6.) # to handle cases with low to medium PA activating range
            except:
                width=so.brentq(objfn1,10**5,10**12) # To handle cases with very high PA activating range 
            return Droot,foldchange,width
        else:
            return 0,0,0
    else:
        return 0,0,0
    
def gendat(paramarr,datax,fitflag=None):
    """Generate sample data point with input paramterarray as list of lists, datax as a list of drug concentrations desired and fitflag to choose Kdim and RAF to be constant as desired. Default: vary all params"""
    if fitflag == None:
        fitflag=0
    
    parset=dict()
    if fitflag == 0:
        parset['Kdim']=paramarr[0]
        parset['RAF']=paramarr[1]
        parset['KA']=paramarr[2]
        miniter=3
    elif fitflag == 1:
        parset['Kdim']=params0['Kdim']
        parset['RAF']=paramarr[0]
        parset['KA']=paramarr[1]
        miniter=2
    elif fitflag == 2:
        parset['Kdim']=params0['Kdim']
        parset['RAF']=params0['RAF']
        parset['KA']=paramarr[0]
        miniter=1
    outy=[]
    for i in range(miniter,len(paramarr)-2,3):
        parset['f']=paramarr[i]
        parset['g']=paramarr[i+1]
        parset['Kd']=paramarr[i+2]
        outy=outy+[[DTOT2AKnorm(dval,parset) for dval in datax]]
    return outy
    
def parsepar(paramarr,fn,fitflag=None,options0=None):
    """applies function fn to each drug parameter combination in the input array: [RAF,KA,f1,g1,Kd1,f2,g1,Kd2,..] """
    if fitflag == None:
        fitflag='0'
    if options0 == None:
        options0=dict()
        options0['KA']=0.0001
        options0['RAFval']=0.04
        options0['Kdimval']=0.1

    parset=dict()
    if fitflag=='0':
        parset['Kdim']=paramarr[0]
        parset['RAF']=paramarr[1]
        parset['KA']=paramarr[2]
        miniter=3
        outy=[]
        for i in range(miniter,len(paramarr)-2,3):
            parset['f']=paramarr[i]
            parset['g']=paramarr[i+1]
            parset['Kd']=paramarr[i+2]
            outy=outy+[fn(parset)]
    elif fitflag=='1':
        parset['Kdim']=options0['Kdimval']
        parset['RAF']=paramarr[0]
        parset['KA']=paramarr[1]
        miniter=2
        outy=[]
        for i in range(miniter,len(paramarr)-2,3):
            parset['f']=paramarr[i]
            parset['g']=paramarr[i+1]
            parset['Kd']=paramarr[i+2]
            outy=outy+[fn(parset)]
    elif fitflag=='2':
        parset['Kdim']=options0['Kdimval']
        parset['RAF']=options0['RAFval']
        parset['KA']=paramarr[0]
        miniter=1
        outy=[]
        for i in range(miniter,len(paramarr)-2,3):
            parset['f']=paramarr[i]
            parset['g']=paramarr[i+1]
            parset['Kd']=paramarr[i+2]
            outy=outy+[fn(parset)]

    elif fitflag == 'DPNC':
        parset['Kdim']=options0['Kdimval']
        parset['RAF']=options0['RAFval']
        parset['KA']=options0['KA'] # KA chosen to be small ~10**-4x
        miniter=0
        outy=[]
        for i in range(miniter,len(paramarr)-2,3):
            parset['f']=paramarr[i]
            parset['g']=paramarr[i+1]
            parset['Kd']=paramarr[i+2]
            outy=outy+[fn(parset)]
    elif fitflag == 'CADP':
        parset['Kdim']=options0['Kdimval']
        parset['RAF']=options0['RAFval']
        parset['KA']=paramarr[0]
        parset['g']=1
        miniter=1
        outy=[]
        for i in range(miniter,len(paramarr)-1,2):
            parset['f']=paramarr[i]
            parset['Kd']=paramarr[i+1]
            outy=outy+[fn(parset)]
    elif fitflag == 'CANC':
        parset['Kdim']=options0['Kdimval']
        parset['RAF']=options0['RAFval']
        parset['KA']=paramarr[0]
        parset['f']=1
        miniter=1
        outy=[]
        for i in range(miniter,len(paramarr)-1,2):
            parset['g']=paramarr[i]
            parset['Kd']=paramarr[i+1]
            outy=outy+[fn(parset)]

    elif fitflag == 'CA':
        parset['Kdim']=options0['Kdimval']
        parset['RAF']=options0['RAFval']
        parset['KA']=paramarr[0]
        parset['f']=1
        parset['g']=1
        miniter=1
        outy=[]
        for i in range(miniter,len(paramarr),1):
            parset['Kd']=paramarr[i]
            outy=outy+[fn(parset)]
    elif fitflag == 'NC':
        parset['Kdim']=options0['Kdimval']
        parset['RAF']=options0['RAFval']
        parset['KA']=0.0001 # KA=10**-4 : ie at equilibrium, the inactive state of RAF is 10**-4x active state
        parset['f']=1
        miniter=0
        outy=[]
        for i in range(miniter,len(paramarr)-1,2):
            parset['g']=paramarr[i]
            parset['Kd']=paramarr[i+1]
            outy=outy+[fn(parset)]
    elif fitflag == 'DP':
        parset['Kdim']=options0['Kdimval']
        parset['RAF']=options0['RAFval']
        parset['KA']=0.0001 # KA chosen to be small ~10**-4x
        parset['g']=1
        miniter=0
        outy=[]
        for i in range(miniter,len(paramarr)-1,2):
            parset['f']=paramarr[i]
            parset['Kd']=paramarr[i+1]
            outy=outy+[fn(parset)]
    parset['rafr']=parset['RAF']/parset['Kdim']

    return outy

#Verify above functions
#actkinr(2.1,params0),actkin(0.21,params0), d2DTOT(0.21,params0),DTOT2d(0.228689,params0),DTOT2AK(0.228689,params0)
#DTOT2AKnorm(0.228689,params0),DTOT2AK(0.228689,params0)/actkin(0,params0)