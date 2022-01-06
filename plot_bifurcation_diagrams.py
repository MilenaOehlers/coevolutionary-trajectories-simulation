import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

steps=5
cut=0.0020
max=cut+0.25
min=-0.25
n=2
cmap = plt.get_cmap('Blues_r')#, cut*2+1) #mycolorbar1 #plt.cm.Blues
cmap.set_bad(color='lightgray')

a= np.load('newnewreldiv_wLi0to10_mig0to1_aPiScale0-5_aWiScale1_relLi0-33to0-67_relPi0-67to0-33_steps5_tfinal1200.npy', encoding='latin1')
plt.figure()
plt.imshow(a[:,:,0],  cmap=cmap, extent=(0,10,0,1), aspect=(10-0)/(1-0))
plt.figure()
plt.imshow(a[:,:,1],  cmap=cmap, extent=(0,10,0,1), aspect=(10-0)/(1-0))
plt.figure()
plt.imshow(a[:,:,2],  cmap=cmap, extent=(0,10,0,1), aspect=(10-0)/(1-0))
plt.show()


fig1, ((pic00, pic01, pic02),(pic10,pic11,pic12),(pic20,pic21,pic22)) = plt.subplots(3, 3)
plt.subplots_adjust(wspace=0, hspace=0)

axlist = [pic00,pic01,pic10,pic11,pic20,pic21]
mycolorbar1 = LinearSegmentedColormap.from_list('mycolorbar1', ['cyan','blue'])#0000AA'])
plt.register_cmap(cmap=mycolorbar1)

pic22.axis('off')
pic21.axis('off')
pic12.axis('off')

cmap = plt.get_cmap('Blues_r')#, cut*2+1) #mycolorbar1 #plt.cm.Blues
cmap.set_bad(color='lightgray')

a=np.zeros((steps,steps))# cmap.set_bad(color='red')
b = np.load('heatmap_reldiv_wLi0to10_mig0to1_aPiScale0-5_aWiScale1_relLi0-33to0-67_relPi0-67to0-33_steps101_tfinal1200.npy', encoding='latin1')  # , encoding='latin1')
for i in range(0,steps):
    for j in range(0,steps):
        help=b[0,i,j,0] if b[0,i,j,0][0]>1000 else b[0,i,j,0][1:]
        #print(b[0, i, j, 0])
        #print(help)
        help=help.astype(float)
        a[i,j]=np.sum((help[1:]-help[0:-1])**(-1))/(len(help)-1) # durchschnittlcihe gemessene frequenz
a = np.ma.masked_where(a <= 0, a)
p0=pic20.imshow(a,  cmap=cmap, extent=(0,10,0,1), aspect=(10-0)/(1-0))
pic20.set_xlabel('$\hat{w_L}$')
pic20.set_ylabel('$\hat{m}$')
pic20.axvline(x=5, color='red',linestyle=':')
pic20.axhline(y=0.5, color='red',linestyle=':')

a=np.zeros((steps,steps))# cmap.set_bad(color='red')
b = np.load('heatmap_reldiv_wLi0to10_migScale0-5_aPi0to1_aWiScale1_relLi0-33to0-67_relPi0-67to0-33_steps101_tfinal1200.npy', encoding='latin1')  # , encoding='latin1')
for i in range(0,steps):
    for j in range(0,steps):
        help=b[0,i,j,0] if b[0,i,j,0][0]>1000 else b[0,i,j,0][1:]
        #print(b[0, i, j, 0])
        #print(help)
        help=help.astype(float)
        a[i,j]=np.sum((help[1:]-help[0:-1])**(-1))/(len(help)-1)
a = np.ma.masked_where(a <= 0, a)
a = np.ma.masked_where(a >cut, a)
p1=pic10.imshow(a,  cmap=cmap, extent=(0,10,0,1), aspect=(10-0)/(1-0))
pic10.axes.get_xaxis().set_ticks([])#.set_xlabel('$\hat{w_L}$')
pic10.set_ylabel('$\hat{\mu_P}$')
pic10.axvline(x=5, color='red',linestyle=':')
pic10.axhline(y=0.5, color='red',linestyle=':')

a=np.zeros((steps,steps))# cmap.set_bad(color='red')
b = np.load('heatmap_reldiv_wLi0to10_migScale0-5_aPiScale0-5_aWi0to10_relLi0-33to0-67_relPi0-67to0-33_steps101_tfinal1200.npy', encoding='latin1')  # , encoding='latin1')
for i in range(0,steps):
    for j in range(0,steps):
        help=b[0,i,j,0] if b[0,i,j,0][0]>1000 else b[0,i,j,0][1:]
        help=help.astype(float)
        a[i,j]=np.sum((help[1:]-help[0:-1])**(-1))/(len(help)-1)
a = np.ma.masked_where(a <= 0, a)
a = np.ma.masked_where(a >cut, a)
p2=pic00.imshow(a,  cmap=cmap, extent=(0,10,0,10), aspect=(10-0)/(10-0))
pic00.axes.get_xaxis().set_ticks([])#.set_xlabel('$\hat{w_L}$')
pic00.set_ylabel('$\hat{\mu_W}$')
pic00.axvline(x=5, color='red',linestyle=':')
pic00.axhline(y=1, color='red',linestyle=':')

a=np.zeros((steps,steps))# cmap.set_bad(color='red')
b = np.load('heatmap_reldiv_wLiScale5_mig0to1_aPi0to1_aWiScale1_relLi0-33to0-67_relPi0-67to0-33_steps101_tfinal1200.npy', encoding='latin1')  # , encoding='latin1')
for i in range(0,steps):
    for j in range(0,steps):
        help=b[0,i,j,0] if b[0,i,j,0][0]>1000 else b[0,i,j,0][1:]
        help=help.astype(float)
        a[i,j]=np.sum((help[1:]-help[0:-1])**(-1))/(len(help)-1)
a = np.ma.masked_where(a <= 0, a)
a = np.ma.masked_where(a >cut, a)
p3=pic11.imshow(a,  cmap=cmap, extent=(0,1,0,1), aspect=(1-0)/(1-0))
pic11.set_xlabel('$\hat{m}$')
pic11.axes.get_yaxis().set_ticks([])
pic11.axvline(x=0.5, color='red',linestyle=':')
pic11.axhline(y=0.5, color='red',linestyle=':')

a=np.zeros((steps,steps))# cmap.set_bad(color='red')
b = np.load('heatmap_reldiv_wLiScale5_mig0to1_aPiScale0-5_aWi0to10_relLi0-33to0-67_relPi0-67to0-33_steps101_tfinal1200.npy', encoding='latin1')  # , encoding='latin1')
for i in range(0,steps):
    for j in range(0,steps):
        help=b[0,i,j,0] if b[0,i,j,0][0]>1000 else b[0,i,j,0][1:]
        help=help.astype(float)
        a[i,j]=np.sum((help[1:]-help[0:-1])**(-1))/(len(help)-1)
a = np.ma.masked_where(a <= 0, a)
a = np.ma.masked_where(a >cut, a)
p4=pic01.imshow(a,  cmap=cmap, extent=(0,1,0,10), aspect=(1-0)/(10-0)) #vmin=min, vmax=max,
pic01.axes.get_xaxis().set_ticks([])#.set_xlabel('$\hat{m}$')
pic01.axes.get_yaxis().set_ticks([])#.set_ylabel('$\hat{\mu_W}$')
pic01.axvline(x=0.5, color='red',linestyle=':')
pic01.axhline(y=1, color='red',linestyle=':')

a=np.zeros((steps,steps))# cmap.set_bad(color='red')
b = np.load('heatmap_reldiv_wLiScale5_migScale0-5_aPi0to1_aWi0to10_relLi0-33to0-67_relPi0-67to0-33_steps101_tfinal1200.npy', encoding='latin1')  # , encoding='latin1')
for i in range(0,steps):
    for j in range(0,steps):
        help=b[0,i,j,0] if b[0,i,j,0][0]>1000 else b[0,i,j,0][1:]
        help=help.astype(float)
        a[i,j]=np.sum((help[1:]-help[0:-1])**(-1))/(len(help)-1)
a = np.ma.masked_where(a <= 0, a)
a = np.ma.masked_where(a >cut, a)
p5=pic02.imshow(a,  cmap=cmap, extent=(0,1,0,10), aspect=(1-0)/(10-0))
pic02.set_xlabel('$\hat{\mu_P}$')
pic02.axes.get_yaxis().set_ticks([])
pic02.axvline(x=0.5, color='red',linestyle=':')
pic02.axhline(y=1, color='red',linestyle=':')

fig1.subplots_adjust(right=0.8)

cbar_ax = fig1.add_axes([0.85, 0.15, 0.045, 0.7])
fig1.colorbar(p1, ax=axlist, cax=cbar_ax)

plt.show()
