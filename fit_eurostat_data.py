import csv
from numpy import *
from pylab import *
import scipy.optimize as optimization

f= open(r'C:\Users\Milu\Documents\BA_Physik\Python_Skripts\xdata.csv', 'r')
r=csv.reader(f, delimiter=';')
xdata=[]
for row in r :
    xdata.append(row)
xdata=array(xdata).astype(float)
f.close()
f= open(r'C:\Users\Milu\Documents\BA_Physik\Python_Skripts\xnames.csv', 'r')
r=csv.reader(f, delimiter=';')
xnames=[]
for [row] in r:
    xnames.append(row)
f.close()

a=zeros((shape(xdata)[0]**2,3))
b=zeros((shape(xdata)[0]**2,3))
fromatob= []
for i in range(0, shape(xdata)[0]):
    for j in range(0,shape(xdata)[0]):
        a[i*shape(xdata)[0]+j,:]=xdata[i,:] # erstelle liste [belgien, belgien,...
        b[i*shape(xdata)[0]+j,:]=xdata[j,:] # liste belgien, deutsch, ..., belgien, deutsch
        fromatob.append([xnames[i],'to',xnames[j]])
datanames=['percapinc', 'pop', 'unemploy']


popvar=zeros((shape(xdata)[0]**2,1))
gdpdiff=zeros((shape(xdata)[0]**2,1))

for i in range(0, shape(xdata)[0]**2):
    popvar[i]=a[i,1]*b[i,1]/(a[i,1]+b[i,1])
    gdpdiff[i]=(a[i,0]-b[i,0])/(a[i,0]+b[i,0]) # so, wie es sein soll.. NACH bel, aus bel, NACH bel, aus deu...

#mymatrix= concatenate((list([['from i', 'to', 'j','popvar','gdpdiff']]), concatenate((fromatob, popvar, gdpdiff), axis=1)), axis=0)
mywork=concatenate((popvar,gdpdiff), axis=1)
#lowrymatrix= concatenate((list([['from i', 'to', 'j', 'Pop i', 'Pop j', 'GDP i', 'GDP j']]), concatenate((fromatob, a[:,1].reshape((900,1)),b[:,1].reshape((900,1)), a[:,0].reshape((900,1)), b[:,0].reshape((900,1))), axis=1)), axis=0)
lowrywork= concatenate((a[:,1].reshape((900,1)),b[:,1].reshape((900,1)), a[:,0].reshape((900,1)), b[:,0].reshape((900,1))), axis=1)

def myfunc(x, mig, pop, gdp):
    y=mig*x[:,0]**pop*x[:,1]**gdp
    y[(x[:,1]<0)]=-mig*x[:,0][(x[:,1]<0)]**pop*(-x[:,1][(x[:,1]<0)])**gdp
    return y

def myfunclog(x, mig, pop, gdp):
    logy = log(mig) + pop * x[:, 0] + gdp * x[:, 1]
    return logy

def myfunc_pop(x, mig, gdp):
    y=mig*x[:,0]*x[:,1]**gdp
    y[(x[:,1]<0)]=-mig*x[:,0][(x[:,1]<0)]*(-x[:,1][(x[:,1]<0)])**gdp
    return y

def myfunc_poplog(x, mig, gdp):
    logy = log(mig) + x[:, 0] + gdp * x[:, 1]
    return logy

def myfunc_gdp(x, mig, pop):
    y=mig*x[:,0]*x[:,1]
    y[(x[:,1]<0)]=-mig*x[:,0][(x[:,1]<0)]**pop*(-x[:,1][(x[:,1]<0)])
    return y

def myfunc_gdplog(x, mig, pop):
    logy = log(mig) + pop*x[:, 0] + x[:, 1]
    return logy

def lowryfunc(x,mig,pop,gdp):
    y = mig * (x[:, 0] * x[:, 1]) ** pop * (x[:, 2] / x[:, 3]) ** gdp #x_3= belg, deu,... x_2=belg, belg...
    return y
    # aber es soll doch NACH bel, aus bel, NACH bel, aus deu sein- jetzt nach bel, AUS bel, nach deu, AUS bel!
    # ^GEÄNDERT, auch in allen anderen lowry funktionen

def lowryfunclog(x, mig, pop, gdp):
    logy= log(mig) + pop*x[:,0]+ pop*x[:,1]+ gdp*(x[:,2]-x[:,3])
    return logy

def lowry2(x,mig,pop,gdp):
    y=mig*(x[:,0]*x[:,1]/(x[:,0]+x[:,1]))**pop*(x[:,2]/x[:,3])**gdp
    return y

def lowrylog2usenonlog(x, mig, pop, gdp):
    logy= log(mig) + pop*log(x[:,0]*x[:,1]/(x[:,0]+x[:,1]))+ gdp*log(x[:,2]/x[:,3])
    return logy

def lowryfree(x, mig, popi, popj, gdpi, gdpj):
    y = mig * x[:, 0]**popi * x[:, 1] ** popj * x[:, 2]**gdpj / x[:, 3] ** gdpi
    return y

def lowryfreelog(x, mig, popi, popj, gdpi, gdpj):
    y = log(mig) + x[:, 0]*popi + x[:, 1]* popj + x[:, 2]*gdpj - x[:, 3]* gdpi
    return y

def lowrypop(x, mig, gdp):
    y= mig * (x[:,0]*x[:,1])**0.5*(x[:,2]/x[:,3])**gdp
    return y

def lowrypoplog(x,mig, gdp):
    logy= log(mig)+ 0.5*(x[:,0]+x[:,1])+gdp*(x[:,2]-x[:,3])
    return logy

def lowrygdp(x, mig, pop):
    y= mig * (x[:,0]*x[:,1])**pop*(x[:,2]/x[:,3])
    return y

def lowrygdplog(x,mig, pop):
    logy= log(mig)+ pop*(x[:,0]+x[:,1])+ (x[:,2]-x[:,3])
    return logy

f=open(r'C:\Users\Milu\Documents\BA_Physik\Python_Skripts\migration_zeilen_nach_spalten_aus.csv') # dateiname genau falsch herum!
r=csv.reader(f, delimiter=';')
migmatrix=[]
for row in r :
    migmatrix.append(row)
migmatrix=array(migmatrix).transpose() # NUN ist in MIGMATRIX Zeilen: Einwanderungsland, Spalten: Auswanderungsland
migmatrix[(migmatrix==':')]=nan
migmatrix=migmatrix.astype(float)

f.close()
lowryvec=migmatrix.reshape((shape(migmatrix)[0]**2,1)) # migij = NETTO wanderung VON J NACH I ! und nicht andersrum! fuck...
print(shape(migmatrix), shape(migmatrix.transpose()))
migij=migmatrix-migmatrix.transpose()
myvec=migij.reshape((shape(migij)[0]**2,1)) # migij = NETTO wanderung VON J NACH I ! und nicht andersrum! fuck...
# reshaped: NACH bel aus bel, NACH bel aus deu, NACH bel aus bul....
my=concatenate((mywork, myvec),axis=1)
my=my[~isnan(my).any(axis=1)]
my=my[~isinf(my).any(axis=1)]
original=len(my[:,0])/2
mylog=log(my[(my[:,1]>0),:])
mylog=mylog[~isnan(mylog).any(axis=1)]
mylog=mylog[~isinf(mylog).any(axis=1)]
remaining=len(mylog[:,0])

lowr=concatenate((lowrywork, lowryvec), axis=1)
lowr=lowr[~isnan(lowr).any(axis=1)]
lowr=lowr[~isinf(lowr).any(axis=1)]
lowrlog=log(lowr)
lowr=lowr[(~isnan(lowrlog).any(axis=1))&(~isinf(lowrlog).any(axis=1))]
lowrlog=lowrlog[~isnan(lowrlog).any(axis=1)]
lowrlog=lowrlog[~isinf(lowrlog).any(axis=1)]


myx=my[:,0:2]
myy=my[:,2]
mylogx=mylog[:,0:2]
mylogy=mylog[:,2]

lowrx=lowr[:,0:4]
lowry=lowr[:,4]
lowrlogx=lowrlog[:,0:4]
lowrlogy=lowrlog[:,4]

def rsquared(myfunc, myworklog, myveclog):
    popt, pcov = optimization.curve_fit(myfunc, myworklog, myveclog)
    myres = myveclog- myfunc(myworklog, popt[0],popt[1],popt[2])
    ss_myres = sum(myres**2)
    ss_tot = sum((myveclog-mean(myveclog))**2)
    r_squared = 1 - (ss_myres / ss_tot)
    return r_squared

def rsquared2(myfunc, myworklog, myveclog):
    popt, pcov = optimization.curve_fit(myfunc, myworklog, myveclog)
    myres = myveclog- myfunc(myworklog, popt[0],popt[1])
    ss_myres = sum(myres**2)
    ss_tot = sum((myveclog-mean(myveclog))**2)
    r_squared = 1 - (ss_myres / ss_tot)
    return r_squared

def rsquared3(myfunc, myworklog, myveclog):
    popt, pcov = optimization.curve_fit(myfunc, myworklog, myveclog)
    myres = myveclog- myfunc(myworklog, popt)
    ss_myres = sum(myres**2)
    ss_tot = sum((myveclog-mean(myveclog))**2)
    r_squared = 1 - (ss_myres / ss_tot)
    return r_squared

def rsquared4(myfunc, myworklog, myveclog):
    popt, pcov = optimization.curve_fit(myfunc, myworklog, myveclog)
    myres = myveclog- myfunc(myworklog, popt[0],popt[1],popt[2], popt[3], popt[4])
    ss_myres = sum(myres**2)
    ss_tot = sum((myveclog-mean(myveclog))**2)
    r_squared = 1 - (ss_myres / ss_tot)
    return r_squared


#popt, pcov=optimization.curve_fit(myfunc, myworklog, myveclog)
#print(popt)
#print(my)


### Teste verschiedene Modelle:
# die, bei der log-Schätzung negatives R^2 produziert hat, werden auskommentiert

popt, pcov = optimization.curve_fit(myfunc, myx, myy)
poptlog, pcovlog = optimization.curve_fit(myfunclog, mylogx, mylogy)

print('meins: R^2: '+str(rsquared(myfunc, myx, myy)))
print('opt. values: '+str(popt))

print('meins log: R^2: '+str(rsquared(myfunclog, mylogx, mylogy)))
print('opt. values: '+str(poptlog))

popt, pcov = optimization.curve_fit(myfunc_pop, myx, myy)
print('meins exponent pop auf 1 fixiert: R^2: ' + str(rsquared2(myfunc_pop, myx, myy)))
print('opt. values: ' + str(popt))

popt, pcov = optimization.curve_fit(myfunc_poplog, mylogx, mylogy)
print('meins exponent pop auf 1 fixiert log: R^2: ' + str(rsquared2(myfunc_poplog, mylogx, mylogy)))
print('opt. values: ' + str(popt))

popt, pcov = optimization.curve_fit(myfunc_gdp, myx, myy)
print('meins exponent gdp auf 1 fixiert: R^2: ' + str(rsquared2(myfunc_gdp, myx, myy)))
print('opt. values: ' + str(popt))

popt, pcov = optimization.curve_fit(myfunc_gdplog, mylogx, mylogy)
print('meins exponent gdp auf 1 fixiert log: R^2: ' + str(rsquared2(myfunc_gdplog, mylogx, mylogy)))
print('opt. values: ' + str(popt))


# popt, pcov = optimization.curve_fit(myfunc_popgdp, myx, myy)
# print('meins exponent pop & gdp auf 1 fixiert: R^2: ' + str(rsquared3(myfunc_popgdp, myx, myy)))
# print('opt. values: ' + str(popt))
#
# popt, pcov = optimization.curve_fit(myfunc_popgdplog, mylogx, mylogy)
# print('meins exponent pop & gdp auf 1 fixiert log: R^2: ' + str(rsquared3(myfunc_popgdplog, mylogx, mylogy)))
# print('opt. values: ' + str(popt))

popt, pcov = optimization.curve_fit(lowryfunc, lowrx, lowry)
print('lowry: R^2: '+str(rsquared(lowryfunc, lowrx, lowry)))
print('opt. values: '+str(popt))

popt, pcov = optimization.curve_fit(lowryfunclog, lowrlogx, lowrlogy)
print('lowry log: R^2: '+str(rsquared(lowryfunclog, lowrlogx, lowrlogy)))
print('opt. values: '+str(popt))

popt, pcov = optimization.curve_fit(lowry2, lowrx, lowrlogy)
print('USE NON LOG for X! USE LOG for Y!')
print('lowry2: R^2: ' + str(rsquared(lowry2, lowrx, lowrlogy)))
print('opt. values: ' + str(popt))

popt, pcov = optimization.curve_fit(lowrylog2usenonlog, lowrx, lowrlogy)
print('lowry2 log: R^2: ' + str(rsquared(lowrylog2usenonlog, lowrx, lowrlogy)))
print('USE NON LOG for X! USE LOG for Y!')
print('opt. values: ' + str(popt))

popt, pcov = optimization.curve_fit(lowryfree, lowrx, lowry)
print('lowryfree: R^2: ' + str(rsquared4(lowryfree, lowrx, lowry)))
print('opt. values: ' + str(popt))

popt, pcov = optimization.curve_fit(lowryfreelog, lowrlogx, lowrlogy)
print('lowryfree log: R^2: ' + str(rsquared4(lowryfreelog, lowrlogx, lowrlogy)))
print('opt. values: ' + str(popt))

popt, pcov = optimization.curve_fit(lowrypop, lowrx, lowry)
print('lowrypop: R^2: ' + str(rsquared2(lowrypop, lowrx, lowry)))
print('opt. values: ' + str(popt))

popt, pcov = optimization.curve_fit(lowrypoplog, lowrlogx, lowrlogy)
print('lowrypop log: R^2: ' + str(rsquared2(lowrypoplog, lowrlogx, lowrlogy)))
print('opt. values: ' + str(popt))

popt, pcov = optimization.curve_fit(lowrygdp, lowrx, lowry)
print('lowrygdp: R^2: ' + str(rsquared2(lowrygdp, lowrx, lowry)))
print('opt. values: ' + str(popt))

popt, pcov = optimization.curve_fit(lowrygdplog, lowrlogx, lowrlogy)
print('lowrygdp log: R^2: ' + str(rsquared2(lowrygdplog, lowrlogx, lowrlogy)))
print('opt. values: ' + str(popt))

print('percent of data points that are not fittable with log in myfunc: '+str(1-remaining/original))