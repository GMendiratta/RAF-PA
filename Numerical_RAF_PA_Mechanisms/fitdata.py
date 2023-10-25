#helpnotes="""Function name: fitdata(ydata,drugvalues,drugnames,*optionsinp={'fitflag':0,'Kdimval':0.1,'fitalgo':'SLSQP','RAFval':0.04,'x0':random in [1,6],'InitVary':True,'DTOTmin':10**-6})"""
def fitdata(ydata,drugvalues,drugnames,optionsinp=None):
    """This function imports the data, drugnames and outputs a dataframe of fit paramteres. ydata=(pMEK/MEK)/(pMEK/MEK)|_d->0 i.e. RAF activation data is assumed normalized to 0 drug and input as a list of lists for n-drugs (including for n=1).\n
    Optional arguments are provided as a dictionary and include,\n 'fitalgo':'SLSQP' accepts any fittingalgorithm for scipy.optimize minimize.\n
    'x0':specifies a fixed list of initial condition value for each parameter. Default choice is a ranfomly chosen value from 0 to 5 for all parameters.\n
    'InitVary': True. Varies initial conditions randomly within boundary conditions if fit is not successful. max iterations is 5.
    'fitflag':0. Default value, '0' varies all parameters. Fitflag '1' specifies Kdim =optionsinp['Kdimval']. '2' specifies Kdim = optionsinp['Kdimval'] and optionsinp['RAFval'].\n
    \n 'Kdimval':0.1 micromolar if unspecified in fitflag scheme '1' or '2'. \n 'DP' represents the drug induced dimerization model. 'NC' represents negative cooperativity. 'CA' represents conformal autoinhibition. 'CANC' drops DP. 'CADP' drops NC. 'DPNC' drops CA.\n\n
    'RAFval':0.04 micromolar if unspecified in fitflag scheme '2'.\n` 
    'Kd_DSFdic':None. Dictionary of minimum and maximum values for Kd for every drug. Format - {'Kd_<drug1>':[min,max],'Kd_<drug2>':[min,max]..},\n
    'DTOTmin':10**-6 micro-molar. minimum value of total drug concentration explored in the fitting.\n
    'KA' key can be specified in optionsinp dictionary and will be used for models without CA mechanism.\n
    'normtype' key can be specified to a value of 0,1,2,3 which represents ={0:1/2,1:1,2:1/4,3:1/6}. The value 1, results in mean absolute % error as the metric. Default value, 0 is better for convergence but is more complex in direct interpretation. This is an advanced option and should be left to default value unless evaluating impact of different norms.
    """

    #BEGIN: -- import libraries --
    import numpy as np
    import scipy.optimize as so
    import pandas as pd
    import random as rand
    #END: -- import libraries --
    
    # check input data
    flag_identicalConcs=True # This flag identifies that all the input drug concentration values are identical for all the drug types. If not, the code automatically discovers this and sets the flag to False.
    if len(ydata) != len(drugnames):
        raise LookupError('ERROR: number of drugs != number of normalized RAF signal lists.')
    elif type(drugvalues[0]) == list:
        flag_identicalConcs=False
        for idrug,iydata in zip(drugvalues,ydata):
            if len(idrug)!=len(iydata):
                raise LookupError('ERROR: Number of input drug concentrations do not match the normalized RAF signal data for at least one drug.')
    elif len(drugvalues) != len(ydata[0]):
        raise LookupError('ERROR: Number of drug concentration points != number of normalized pMEK/MEK points for first drug.')

    #BEGIN -- Initialize Options --
    options0={'fitflag':'0','fitalgo':'SLSQP','InitVary':True,'DTOTmin':10**-6,'Kd_DSFdic':False,'normtype':0} # Kdim,RAF,DTOTmin in micro molar.
    if optionsinp != None:
        for key1 in optionsinp.keys():
            options0[key1]=optionsinp[key1] # update options with input values
    if (((options0['fitflag'] == '1') or (options0['fitflag'] == '2')) and ('Kdimval' not in options0.keys())):
        options0['Kdimval']=0.1 # default Kdim if not specified
    if ((options0['fitflag'] == '2') and ('RAFval' not in options0.keys())):
        options0['RAFval']=0.04 # default RAF if not specified
    if ('KA' not in options0.keys()):
        options0['KA']=0.0001 # No CA mechanism if KA not specified in DPNC, DP and NC models
    # print("Options choice: ",options0) # uncomment this line to track the progress of fitting. - slows down code for i/o.
    #END -- Initialize Options -- 
    
    # global boundary values
    bdyglobal={'f':[10**-5,100],'g':[1,10**4],'KA':[0.001,10**2],'Kd':[10**-4,10**4],'Kdim':[10**-4,10**4],'RAF':[10**-4,10**3]} # Units: micro-Molar
    DTOTmin=options0['DTOTmin']
    
    ## ---BEGIN: Sub-functions --- 
    def checkpos(x):
        """Returns 'True' if all elements in a list are positive numbers greater than 10^-7 else False"""
        flag=True
        if type(x) == dict:
            x=list(x.values())
        for elem in x:
            if elem<=10**-7:
                flag=False
        return flag
        
    def checkbdy(x0,bdy):
        """Check if x0 lies within respective boundary values provided in bdy."""
        flag=True
        if len(x0)!=len(bdy):
            flag=False
            print('ERROR: x0 and bounds vectors of different length.')
        try:
            for i in range(len(x0)):
                if (x0[i]<bdy[i][0]) or (x0[i]>bdy[i][1]):
                    flag=False
        except:
            flag=False
        return flag
    
    def randchoose(bdy):
        """This function inputs a list of two numbers and outputs a random numbers log10-uniformly in the range an order of magnitude reduced on each side. Note that this function should only be used to identify initial conditions based on the boundries. It should not be used in an MC search for best fit as the full boundary is not sampled due to convergence issues common at the boundary values."""
#         return [1+5*rand.random()]*len(bdy) # an alternative randchoose method where convergence is FAR better.
        outlist=[]
        for inparr in bdy:
            min0=np.log10(inparr[0])+1
            max0=np.log10(inparr[1])-1
            diff0=abs(max0-min0)
            randno=rand.random()
            randlog=min0+diff0*randno
            outlist=outlist+[10**randlog]
        return outlist

    def arrcompare(x1,x2):
        """returns the relative difference between corresponding values in lists x1 and x2. """
        try:
            res=[]
            for i in range(len(x1)):
                res=res+[(x1[i]-x2[i])/x1[i]]
            return res
        except: return 'ERROR'

    #BEGIN: -- Model Functions --
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
        try:
            return (d*(8*(d**2 + 2*d*params['g']*params['Kd'] + params['f']*params['g']*params['Kd']**2)**2 - 2*params['f']*params['g']*(d**2 + 2*d*params['g']*params['Kd'] + params['f']*params['g']*params['Kd']**2)*params['Kdim']*(d + params['Kd'] + params['KA']*params['Kd'] - np.sqrt((d + params['Kd'] + params['KA']*params['Kd'])**2 + (8*(d**2 + 2*d*params['g']*params['Kd'] + params['f']*params['g']*params['Kd']**2)*params['RAF'])/(params['f']*params['g']*params['Kdim']))) + d*params['f']*params['g']*params['Kdim']*(d + params['Kd'] + params['KA']*params['Kd'] - np.sqrt((d + params['Kd'] + params['KA']*params['Kd'])**2 + (8*(d**2 + 2*d*params['g']*params['Kd'] + params['f']*params['g']*params['Kd']**2)*params['RAF'])/(params['f']*params['g']*params['Kdim'])))**2 + params['f']*params['g']**2*params['Kd']*params['Kdim']*(d + params['Kd'] + params['KA']*params['Kd'] - np.sqrt((d + params['Kd'] + params['KA']*params['Kd'])**2 + (8*(d**2 + 2*d*params['g']*params['Kd'] + params['f']*params['g']*params['Kd']**2)*params['RAF'])/(params['f']*params['g']*params['Kdim'])))**2))/(8*(d**2 + 2*d*params['g']*params['Kd'] + params['f']*params['g']*params['Kd']**2)**2)
        except:
            print("ERROR: d2DTOT: ",d,params)

    def DTOT2d(DTOT,params):
        """Numerically solves the inverse function d2DTOT to convert input total drug concentration and parameters into unbound drug conccentration"""
        try:
            DTOT=int(DTOT) if DTOT>10**8 else DTOT # for very large numbers, the floating point is irrelevant to precision but crashes optimization algorithms.
            if DTOT>DTOTmin and checkpos(list(params.values())):
                objfn=lambda d:(DTOT-d2DTOT(d,params))/DTOT
                return so.brentq(objfn,10.**-14,DTOT,maxiter=100)
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
    #END: -- Model Functions --

    
    #BEGIN: -- Fitting Algorithms  --
    normdic={0:1/2,1:1,2:1/4,3:1/6} # defined as 1/p-norm
    
    def metricfn(fn,parset,xdat,ydat,normtype=None):
        """Inputs the weight function, parameter dictionary,domain values array, range/data values array, norm-metric type.
        Each x value and the parameters dictionary is fed to weight function to get predicted data.
        Specified normtype (0=L2,2=L4) is used relative to input ydat (range values) to calculate the total, summed objective scalar function."""
        if normtype == None:
            normtype=0 #L2 normalized is the default norm
        obj=0.
        for i in range(len(ydat)):
            yval=fn(xdat[i],parset)
            obj=obj+abs((yval-ydat[i])/ydat[i])**(1/normdic[normtype])
        return obj
    
    def objfunc(paramarr,xdat,ydat,normtype=None,scaling=None,fitflag=None):
        """This function inputs a parameter set, the drug data and corresponding normalized activity levels. It outputs a combined objective function calculated using the model predictions of the parameter sn defined local objective function over all the input drugs.
        The input paramarr has to arranged as : [Kdim,RAF,KA,f1,g1,Kd1,f2,g2,kd2..]"""
        if fitflag == None:
            fitflag='0'
        if normtype == None:
            normtype=0 #abs deviation norm
        if scaling == None:
            scaling=1 # simply to prevent the code from breaking.
        if not checkpos(paramarr):
            return 10**10

        parset=dict()
        if ((fitflag=='0') or (fitflag=='1') or (fitflag=='2')):
            if fitflag=='0':
                parset['Kdim']=paramarr[0]
                parset['RAF']=paramarr[1]
                parset['KA']=paramarr[2]
                miniter=3
            elif fitflag=='1':
                parset['Kdim']=options0['Kdimval']
                parset['RAF']=paramarr[0]
                parset['KA']=paramarr[1]
                miniter=2
            elif fitflag=='2':
                parset['Kdim']=options0['Kdimval']
                parset['RAF']=options0['RAFval']
                parset['KA']=paramarr[0]
                miniter=1
            obfn=0.
            j=0 # tracks the drug 
            for i in range(miniter,len(paramarr)-2,3):
                parset['f']=paramarr[i]
                parset['g']=paramarr[i+1]
                parset['Kd']=paramarr[i+2]
                try:
                    if flag_identicalConcs:
                        obfn=obfn+metricfn(DTOT2AKnorm,parset,xdat,ydat[j],normtype)
                    elif ~flag_identicalConcs:
                        obfn=obfn+metricfn(DTOT2AKnorm,parset,xdat[j],ydat[j],normtype)
                except:
                    # in case any error is raised for the input values of parameters, the objective function is set to 10^10 thereby making this paramter unviable in any type of minimization.
                    print("ERROR obfunc",[metricfn(DTOT2AKnorm,parset,xdat,ydat[j],normtype),parset,xdat,ydat[j],normtype])
                    obfn=10**10
                    return obfn
                j=j+1

        elif fitflag == 'DPNC':
            parset['Kdim']=options0['Kdimval']
            parset['RAF']=options0['RAFval']
            parset['KA']=options0['KA']
            obfn=0.
            j=0 # tracks the drug 
            miniter=0
            for i in range(miniter,len(paramarr)-2,3):
                parset['f']=paramarr[i]
                parset['g']=paramarr[i+1]
                parset['Kd']=paramarr[i+2]
                try:
                    if flag_identicalConcs:
                        obfn=obfn+metricfn(DTOT2AKnorm,parset,xdat,ydat[j],normtype)
                    elif ~flag_identicalConcs:
                        obfn=obfn+metricfn(DTOT2AKnorm,parset,xdat[j],ydat[j],normtype)
                except:
                    # in case any error is raised for the input values of parameters, the objective function is set to 10^10 thereby making this paramter unviable in any type of minimization.
                    print("ERROR obfunc",[metricfn(DTOT2AKnorm,parset,xdat,ydat[j],normtype),parset,xdat,ydat[j],normtype])
                    obfn=10**10
                    return obfn
                j=j+1
            
        
        elif fitflag == 'CADP':
            parset['Kdim']=options0['Kdimval']
            parset['RAF']=options0['RAFval']
            parset['KA']=paramarr[0]
            parset['g']=1
            obfn=0.
            j=0 # tracks the drug
            miniter=1
            for i in range(miniter,len(paramarr)-1,2):
                parset['f']=paramarr[i]
                parset['Kd']=paramarr[i+1]
                try:
                    if flag_identicalConcs:
                        obfn=obfn+metricfn(DTOT2AKnorm,parset,xdat,ydat[j],normtype)
                    elif ~flag_identicalConcs:
                        obfn=obfn+metricfn(DTOT2AKnorm,parset,xdat[j],ydat[j],normtype)
                except:
                    # in case any error is raised for the input values of parameters, the objective function is set to 10^10 thereby making this paramter unviable in any type of minimization.
                    print("ERROR obfunc",[metricfn(DTOT2AKnorm,parset,xdat,ydat[j],normtype),parset,xdat,ydat[j],normtype])
                    obfn=10**10
                    return obfn
                j=j+1

        elif fitflag == 'CANC':
            parset['Kdim']=options0['Kdimval']
            parset['RAF']=options0['RAFval']
            parset['KA']=paramarr[0]
            parset['f']=1
            obfn=0.
            j=0 # tracks the drug 
            miniter=1
            for i in range(miniter,len(paramarr)-1,2):
                parset['g']=paramarr[i]
                parset['Kd']=paramarr[i+1]
                try:
                    if flag_identicalConcs:
                        obfn=obfn+metricfn(DTOT2AKnorm,parset,xdat,ydat[j],normtype)
                    elif ~flag_identicalConcs:
                        obfn=obfn+metricfn(DTOT2AKnorm,parset,xdat[j],ydat[j],normtype)
                except:
                    # in case any error is raised for the input values of parameters, the objective function is set to 10^10 thereby making this paramter unviable in any type of minimization.
                    print("ERROR obfunc",[metricfn(DTOT2AKnorm,parset,xdat,ydat[j],normtype),parset,xdat,ydat[j],normtype])
                    obfn=10**10
                    return obfn
                j=j+1

        elif fitflag == 'DP':
            parset['Kdim']=options0['Kdimval']
            parset['RAF']=options0['RAFval']
            parset['KA']=options0['KA']
            parset['g']=1
            obfn=0.
            j=0 # tracks the drug 
            miniter=0
            for i in range(miniter,len(paramarr)-1,2):
                parset['f']=paramarr[i]
                parset['Kd']=paramarr[i+1]
                try:
                    if flag_identicalConcs:
                        obfn=obfn+metricfn(DTOT2AKnorm,parset,xdat,ydat[j],normtype)
                    elif ~flag_identicalConcs:
                        obfn=obfn+metricfn(DTOT2AKnorm,parset,xdat[j],ydat[j],normtype)
                except:
                    # in case any error is raised for the input values of parameters, the objective function is set to 10^10 thereby making this paramter unviable in any type of minimization.
                    print("ERROR obfunc",[metricfn(DTOT2AKnorm,parset,xdat,ydat[j],normtype),parset,xdat,ydat[j],normtype])
                    obfn=10**10
                    return obfn
                j=j+1

        elif fitflag == 'NC':
            parset['Kdim']=options0['Kdimval']
            parset['RAF']=options0['RAFval']
            parset['KA']=options0['KA']
            parset['f']=1
            obfn=0.
            j=0 # tracks the drug 
            miniter=0
            for i in range(miniter,len(paramarr)-1,2):
                parset['g']=paramarr[i]
                parset['Kd']=paramarr[i+1]
                try:
                    if flag_identicalConcs:
                        obfn=obfn+metricfn(DTOT2AKnorm,parset,xdat,ydat[j],normtype)
                    elif ~flag_identicalConcs:
                        obfn=obfn+metricfn(DTOT2AKnorm,parset,xdat[j],ydat[j],normtype)
                except:
                    # in case any error is raised for the input values of parameters, the objective function is set to 10^10 thereby making this paramter unviable in any type of minimization.
                    print("ERROR obfunc",[metricfn(DTOT2AKnorm,parset,xdat,ydat[j],normtype),parset,xdat,ydat[j],normtype])
                    obfn=10**10
                    return obfn
                j=j+1


        elif fitflag == 'CA':
            parset['Kdim']=options0['Kdimval']
            parset['RAF']=options0['RAFval']
            parset['KA']=paramarr[0]
            parset['f']=1
            parset['g']=1
            obfn=0.
            j=0 # tracks the drug 
            miniter=1
            for i in range(miniter,len(paramarr)-1,1):
                parset['Kd']=paramarr[i]
                try:
                    if flag_identicalConcs:
                        obfn=obfn+metricfn(DTOT2AKnorm,parset,xdat,ydat[j],normtype)
                    elif ~flag_identicalConcs:
                        obfn=obfn+metricfn(DTOT2AKnorm,parset,xdat[j],ydat[j],normtype)
                except:
                    # in case any error is raised for the input values of parameters, the objective function is set to 10^10 thereby making this paramter unviable in any type of minimization.
                    print("ERROR obfunc",[metricfn(DTOT2AKnorm,parset,xdat,ydat[j],normtype),parset,xdat,ydat[j],normtype])
                    obfn=10**10
                    return obfn
                j=j+1
        return 100*(obfn**normdic[normtype])/scaling #<<MAIN return statement>> for objective function. for l2 norm (normtype=0), the 100 factor converts the objective function in % square root of squared deviation ratios (to data).

    def parseparfit(paramarr,fn,fitflag=None):
        """applies function fn to each drug parameter combination in the input array: [RAF,KA,f1,g1,Kd1,f2,g1,Kd2,..] """
        if fitflag == None:
            fitflag='0'
    
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
    #END: -- Fitting Algorithms  --
    ##END: ---Sub-functions --- 
   
    
    #BEGIN: ---MAIN Function ---
    ##  Define Parameters and embed into Objective Function
    xdat0=drugvalues
    ydat0=ydata
    ndrugs=len(ydat0)
    if options0['fitflag'] == '0':
        keylist=['Kdim','RAF','KA']
        for idrug in drugnames:
            addKdstr='_'+idrug if (options0['Kd_DSFdic']and(idrug in options0['Kd_DSFdic'].keys())) else ''
            keylist=keylist+['f','g','Kd'+addKdstr] # drug specific parameters when Kd for each drug are specified.
    elif options0['fitflag'] == '1':
        keylist=['RAF','KA'] 
        for idrug in drugnames:
            addKdstr='_'+idrug if (options0['Kd_DSFdic']and(idrug in options0['Kd_DSFdic'].keys())) else ''
            keylist=keylist+['f','g','Kd'+addKdstr] # drug specific parameters when Kd for each drug are specified.
    elif options0['fitflag'] == '2':
        keylist=['KA'] 
        for idrug in drugnames:
            addKdstr='_'+idrug if (options0['Kd_DSFdic']and(idrug in options0['Kd_DSFdic'].keys())) else ''
            keylist=keylist+['f','g','Kd'+addKdstr] # drug specific parameters when Kd for each drug are specified.

    elif options0['fitflag'] == 'DPNC':
        keylist=[] # Drug induced dimerization + negative cooperativity model. No KA.
        for idrug in drugnames:
            addKdstr='_'+idrug if (options0['Kd_DSFdic'] and (idrug in options0['Kd_DSFdic'].keys())) else ''
            keylist=keylist+['f','g','Kd'+addKdstr] # drug specific parameters when Kd for each drug are specified.
    elif options0['fitflag'] == 'CADP':
        keylist=['KA'] # Reversible conformal autoinhibition + drug induced dimerization. No g.
        for idrug in drugnames:
            keylist=keylist+['f','Kd']
    elif options0['fitflag'] == 'CANC':
        keylist=['KA'] # Reversible conformal autoinhibition + negative cooperativity. No f.
        for idrug in drugnames:
            keylist=keylist+['g','Kd']


    elif options0['fitflag'] == 'NC':
        keylist=[] #negative cooperativity. No f,KA.
        for idrug in drugnames:
            keylist=keylist+['g','Kd']
    elif options0['fitflag'] == 'CA':
        keylist=['KA'] # Reversible conformal autoinhibition. No f,g.
        for idrug in drugnames:
            keylist=keylist+['Kd']
    elif options0['fitflag'] == 'DP':
        keylist=[] # Drug induced dimerization. No KA,g.
        for idrug in drugnames:
            keylist=keylist+['f','Kd']
            
    bdy= [(options0['Kd_DSFdic'][key1] if 'Kd_' in key1 else bdyglobal[key1]) for key1 in keylist] # bounds for each parameter
    # specify random initial conditions as default.
    if 'x0' not in optionsinp.keys():
        options0['x0']=[4*rand.random()+1]*(len(bdy)) # choosing values near order 1 as initial conditions increases likelihood of convergence. values>1 ensure that g>1 constraint is satisfied.

    scaling=sum([len(idat) for idat in ydat0]) #= number of total data points fit in normalized basis (not including d=0).
    objlam= lambda pars:objfunc(pars,xdat0,ydat0,normtype=options0['normtype'],scaling=scaling,fitflag=options0['fitflag']) # objective function L2  norm (normalized to number of drugs)
    try:
        flagsuccess=False
        itry=0
        if checkbdy(options0['x0'],bdy):
            x00=options0['x0']
            # To fix initial condition chosen, turn 'InitVary' to False in the input options.
        elif options0['InitVary']:
            x00=randchoose(bdy)  # if InitVary is set to false - some runs may fail with default initial conditions. Default is True.
        else:
            raise RuntimeError('Initialization failed. Initial value not within boundary conditions. Check boundary conditions.')
            return []
        while (flagsuccess == False) and (itry<5):
            try:
                if options0['fitalgo'] == 'SLSQP':
                    # This is the default option
                    resobj=so.minimize(objlam,x0=x00,bounds=bdy,method='SLSQP',options={'maxiter':300})
                    flagsuccess=resobj.success
                elif options0['fitalgo'] == 'L-BFGS-B':
                    resobj=so.minimize(objlam,x0=x00,bounds=bdy,method='L-BFGS-B',options={'maxfun':20000})# default method for constrained minimization is L-BFGS-B
                    flagsuccess=resobj.success
                else:
                    resobj=so.minimize(objlam,x0=x00,bounds=bdy,method=options0['fitalgo'])
                    flagsuccess=resobj.success
                if (not flagsuccess) and options0['InitVary']:
                    # if initial condition is not working try others randomly: I have emperically found that order 1  values lead to better convergence
                    x00=randchoose(bdy)
                    rand.random()
            except:
                # if initial condition is not working try others randomly
                if options0['InitVary'] and (itry<4):
                    x00=randchoose(bdy)
                    rand.random()
                else:
                    print('so.minimize failed for options:',options0)
                    return []
            itry=itry+1
    except:
        print('so.minimize failed for options:',options0)
        return []
    if not resobj.success:
        return []
    elif resobj.success:
        dfout=pd.concat([pd.DataFrame.from_dict([dic1]) for dic1 in parseparfit(resobj.x,lambda x:dict(x),fitflag=options0['fitflag'])])
        dfout['drug']=drugnames
        dfout['fitmetric']=resobj.fun
        dfout['algorithm']=options0['fitalgo']
        dfout['init']=str(x00[0]) if ((x00[0]<5) and (x00[0]>1)) else str(x00)
        return dfout
    #END: -- Main Function --