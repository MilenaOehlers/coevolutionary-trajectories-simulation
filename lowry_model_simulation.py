import mpi # source code: http://www.pik-potsdam.de/~donges/pyunicorn/_modules/pyunicorn/utils/mpi.html
import time
import numpy as np
import itertools as itt
from scipy.integrate import odeint

new=1
n = 2  		# no. of world regions
steps = 3   # Feinheit Bifurkation
st = 0 		# falls werte 0,0 der parameter probleme machen, setze st=1
tfinal = 1200 	# wie lange (zeitlich gesehen) Pi messen?
dt0 = 0.001 	# wie genau Pi messen (Zeitabstaende, je kleiner, desto genauer)
relbifver=1
div=1

### Codedummys: was soll durchgefahren werden?
wLibif = 1
migbif = 1
aPibif = 0
aWibif = 0

### wie sollen nicht durchgefahrene Variablen skaliert werden?
wLirel = 5
migrel = 0.5
aPirel = 0.5
aWirel = 1

# von wo bis wo werden Variablen durchgefahren?
lista = list([[0, 7], # von wLi: start und Endwert
              [0.3, 0.7], # von mig: start und Endwert
              [0.3,0.7], # von aPi: start und Endwert
              [0.3, 3]]) # von aWi: start und Endwert

# relative Anfangsausstattung Regionen:
relLi= np.ones(n)
relPi=np.ones(n)
for i in range(0,n):
    relPi[i] = float(n - i) / (n * (n + 1) / 2)
    relLi[i]=float(i+1)/(n*(n+1)/2)

Cstar = 4000  # 5500  # total carbon in atmosphere, plants/soil, fossils, and upper ocean [GtC]
m = 1.5  # 1.43  # solubility coefficient [1]
mig = 1.  # global migration parameter

a0i = 0.0298 * np.repeat(1., n)  # respiration baseline coefficient [a^-1]
aTi = 3.2e3 * np.repeat(1., n)  # respiration sensitivity to temperature [km^2a^-1GtC^-1]
aPi = 1. * np.repeat(1., n)
aWi = 1. * np.repeat(1., n)
l0i = 26.4 * np.repeat(1., n)  # photosynthesis baseline coefficient [km a^-1GtC^-1/2]
lTi = 1.1e6 * np.repeat(1., n)  # photosynthesis sensitivity to temperature [km^3 a^-1 GtC^-3/2]
pi = 0.04 * np.repeat(1., n)  # fertility rate maximum [a^-1]
Sigmai = np.repeat(1.5e8 / n, n)  # land area [km]
q0i = 20 * np.repeat(1., n)  # mortality baseline coefficient [$a-2 H^-1]
WPi = 2000 * np.repeat(1., n)  # fertility saturation wellbeing [$a^-1H^-1]
wLi = 4.55e7 * np.repeat(1., n)
yBi = 2.47e11 * np.repeat(1., n)  # 1e4#2.47e9
b = 5.4e-7

# globale Anfangsausstattung
x0 = {'Li': 2880 * relLi,  # 2480,  # terrestrial carbon [GtC]
      'Pi': 500000 * relPi,  # human population [humans]
}
#def xrange(a):
#    return range(0,a)

def run(x0, T, dt, wLi0, mig0, aPi0, aWi0):
    def trajectories(xt):
        res = [[val] for val in values(xt[0, :])]
        for i in range(1, xt.shape[0]):
            for j, val in enumerate(values(xt[i, :])):
                res[j].append(val)
        return tuple([np.array(l) for l in res])

    def values(x):
        Li = x[:n]  # carbon in vegetables
        Pi = x[n:2 * n]  # population
        # Bi = x[2*n:3*n] # biomass harvest

        A = (Cstar - sum(Li)) / (m + 1)  # athmosph. carbon
        M = m * A  # maritime carbon
        T = A / sum(Sigmai)  # temperature

        Bi = b * Pi ** 0.6 * Li ** 0.4
        Yi = yBi * Bi  # = min(yEi*Ei, f(Pi,Ki))

        percapinc = Yi / Pi
        percapinc[np.where(Pi==0)[0]]=0
        Wi = percapinc + wLi0 * Li / Sigmai

        Mij= np.zeros((n,n))
        for i in range(0,n):
            for j in range(0,n):
                if i!=j:
                    mijhelp=mig0*Pi[i]**aPi0[i]*Pi[j]**aPi0[j]*Wi[i]**aWi0[i]/Wi[j]**aWi0[j]
                    Mij[i, j] = mijhelp
                else:
                    Mij[i,j]=0
            for j in range(0,n):
                if new==1:
                    Mij[i,j]= Mij[i,j]
                else:
                    Mij[i, j] = Mij[i, j] if sum(Mij[:,j]) < Pi[j] else Mij[i,j]*Pi[j]/sum(Mij[:,j])

        migrationi = np.zeros(n)
        for i in range(0,n):
            migrationi[i] = sum(Mij[i,:])-sum(Mij[:,i])
            if migrationi[i]<(-Pi[i]):
                print('!')

        return A, Bi, Li, M, migrationi, Pi, T, Wi, Yi

    def dlogx_dt(logx, unused_t):
        x = np.exp(logx)

        A, Bi, Li, M, migrationi, Pi, T, Wi, Yi = values(x)

        respiration = Li * (a0i + aTi * T)
        photosynthesis = Li * (l0i - lTi * T) * np.sqrt(A / sum(Sigmai))

        fertility = 2 * pi * Wi * WPi / (Wi ** 2 + WPi ** 2)
        basic_mortality = q0i / Wi
        basic_mortality[np.where(Wi==0)[0]]=0#np.where(np.isnan(basic_mortality) + np.isinf(basic_mortality))[0]] = 0

        dLi_dt = photosynthesis - respiration - Bi
        dPi_dt = Pi * (fertility - basic_mortality) + migrationi

        dx_dt = np.concatenate((dLi_dt, dPi_dt)).reshape((-1,))
        helpx=dx_dt/x
        helpx[np.where(x==0)[0]]=0
        return helpx  # = dlogx_dt

    logx0 = np.log(np.concatenate((x0['Li'], x0['Pi'])))
    t = np.arange(0, T, dt)

    logxt = odeint(dlogx_dt, logx0, t)
    # now logx is a 2d-array with first axis = time index
    xt = np.exp(logxt)

    At, Bti, Lti, Mt, migrationti, Pti, Tt, Wti, Yti = trajectories(xt)

    return {'Pi': Pti,
            'Li': Lti}


vec0 = np.array([wLibif, migbif, aPibif, aWibif])
vec1 = [wLi, mig, aPi, aWi]
vec2 = ['wLi', 'mig', 'aPi', 'aWi']

def master(): # Chef-Funktion
    start_time = time.time()

    for index in range(0,steps):
        mpi.submit_call("loop", (index,relbifver,div,vec0,vec1,vec2), id=index)

    results = {i: mpi.get_result(id=i) for i in range(0,steps-st)}

    resultma=np.zeros((steps,steps,3)) #last: noosci, avefreq, aveampli
    for i in range(0,steps):
        for j in range(0,steps):
            for hh in range(0,3):
                resultma[i,j,hh]=results[i][j,hh]
    for pp in range(0,3):
        help=resultma[:, :, pp]
        resultma[:, :, pp]=help[::-1].transpose()[::-1].transpose()[::-1].transpose()
    

    ewLi = str(lista[0][0]) + 'to' + str(lista[0][1]) if wLibif == 1 else 'Scale' + str(wLirel)
    emig = str(lista[1][0]) + 'to' + str(lista[1][1]) if migbif == 1 else 'Scale' + str(migrel)
    eaPi = str(lista[2][0]) + 'to' + str(lista[2][1]) if aPibif == 1 else 'Scale' + str(aPirel)
    eaWi = str(lista[3][0]) + 'to' + str(lista[3][1]) if aWibif == 1 else 'Scale' + str(aWirel)
    newnew='new' if new==1 else 'old'
    named = newnew+'_wLi' + ewLi + '_mig' + emig + '_aPi' + eaPi + '_aWi' + eaWi + '_relLi' + str(
        relLi[0].round(2)) + 'to' + str(relLi[1].round(2)) + '_relPi' + str(relPi[0].round(2)) + 'to' + str(
        relPi[1].round(2)) + '_steps' + str(steps) + '_tfinal' + str(tfinal)
    np.save('lowlow_'+str(named.replace('.', '-')), resultma)

    print('time elapsed: '+str(time.time()-start_time)+'s')

def loop(index, vec0, vec1, vec2): # Mitarbeiter-Funktion
    i = index
    for k,l in itt.product(range(0, len(vec0)),range(0, len(vec0))): # findet bifur-variablen, die durchfahren werden sollen
        if vec0[k] == 1 and vec0[l] == 1 and k < l:
            scalei = np.linspace(lista[k][0], lista[k][1], steps)
            scalej = np.linspace(lista[l][0], lista[l][1], steps)

            zerovec = np.zeros((n, steps-st, int(tfinal/dt0)))
            xyzeros=np.zeros((n,steps-st,4))
            xyzeros=np.array(xyzeros, dtype=object)
            noosci=np.zeros(steps)
            avefreq=np.zeros(steps)
            aveampl=np.zeros(steps)
            returnma=np.zeros((steps,3))
            for j in range(0, steps):
                wLi0 = vec1[vec2.index('wLi')] * scalei[i] if vec2.index('wLi') == k \
                    else vec1[vec2.index('wLi')] * scalej[j] if vec2.index('wLi') == l \
                    else wLi * wLirel
                mig0 = vec1[vec2.index('mig')] * scalei[i] if vec2.index('mig') == k \
                    else vec1[vec2.index('mig')] * scalej[j] if vec2.index('mig') == l \
                    else mig * migrel
                aPi0 = vec1[vec2.index('aPi')] * scalei[i] if vec2.index('aPi') == k \
                    else vec1[vec2.index('aPi')] * scalej[j] if vec2.index('aPi') == l \
                    else aPi * aPirel
                aWi0 = vec1[vec2.index('aWi')] * scalei[i] if vec2.index('aWi') == k \
                    else vec1[vec2.index('aWi')] * scalej[j] if vec2.index('aWi') == l \
                    else aWi * aWirel
                r = run(x0=x0, T=tfinal, dt=dt0, wLi0=wLi0, mig0=mig0, aPi0=aPi0, aWi0=aWi0)

                ### relbif version start
                bif = np.ones((n, int(tfinal / dt0)))
                lif = np.ones((n, int(tfinal / dt0)))
                for g in range(0,n):
                    bif[g, :] = r['Pi'][:, g]
                    lif[g, :] = r['Li'][:, g]
                avebif = np.sum(bif, axis=0) / n
                avelif = np.sum(lif, axis=0) / n

                relbif = bif[0, :] / avebif[:]
                rellif= lif[0, :] / avelif[:]
                for t in range(0, int(tfinal/dt0)-1):
                    zerovec[0, j, t] = 1 if relbif[t] >= 1 and relbif[t + 1] < 1 else 1 if relbif[t] <= 1 and relbif[t + 1] > 1 else 0
                xyzeros[0, j, 0]=np.where(zerovec[0,j,:]==1)[0]

                amplivec=np.zeros(len(xyzeros[0, j, 0]))

                for blub in range(0,len(xyzeros[0, j, 0])-1):
                    amplivec[blub]=max(abs(relbif[xyzeros[0,j,0][blub]:xyzeros[0,j,0][blub+1]]-1))
                amplivec2 = abs(rellif[xyzeros[0, j, 0]]-1)
                amplivec[len(xyzeros[0, j, 0])-1] = max(abs(relbif[xyzeros[0, j, 0][-1]: -1] - 1))
                xyzeros[0, j, 1] = amplivec
                xyzeros[0,j,2]=amplivec2
                xyzeros[0,j,3]=(avebif[xyzeros[0,j,0]]/avelif[xyzeros[0,j,0]])**0.6
                noosci[j]=len(xyzeros[0, j, 0])
                help = xyzeros[0, j, 0] if xyzeros[0, j, 0][0] > 1000 else xyzeros[0, j, 0][1:]
                help = help.astype(float)
                avefreq[j]=np.sum((help[1:]-help[0:-1])**(-1))/(len(help)-1)
                aveampl[j]=np.sum(xyzeros[0, j, 1][1:].astype(float)/xyzeros[0,j,2][1:].astype(float))/len(xyzeros[0, j, 1][1:])
                returnma[j,0]=noosci[j]
                returnma[j, 1] = avefreq[j]
                returnma[j, 2] = aveampl[j]

    return returnma



mpi.run()

