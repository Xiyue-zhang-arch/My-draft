import numpy as np
from scipy.signal import hilbert
from common import LENGTH,SPR,ORDERS


def reference_one(raw,orders=ORDERS):
    x=raw.astype('float64').copy(); x-=x.mean(-1,keepdims=True)
    active=np.sqrt(np.mean(x*x,axis=-1)); x/=np.maximum(active[:,None],np.finfo(float).tiny)
    window=np.hanning(LENGTH+1)[:-1]; grid=np.fft.rfftfreq(LENGTH,1/SPR)
    def power(x):
        x=x-np.sum(x*window)/window.sum()
        p=np.abs(np.fft.rfft(x*window))**2; p[0]=p[-1]=0
        return p
    def band(current):
        mask=[]
        for base in orders:
            m=np.zeros_like(grid,dtype=bool)
            for h in (1,2):
                center=h*base+(4 if current else 0); width=max(.02*center,.5)
                m|=(grid>=center-width)&(grid<center+width)
            mask.append(m)
        return np.array(mask)
    e=np.abs(hilbert(x[0])); vpower=power(e**2); vb=band(False)
    r=[float(vpower[m].sum()/max(vpower.sum(),np.finfo(float).tiny)) for m in vb]
    u=[]
    for base in orders:
        peaks=[]
        for period in (1,2,3):
            center=SPR*period/base; width=max(1.,.02*center)
            values=[]
            for lag in range(int(np.ceil(center-width)),int(np.floor(center+width))+1):
                a=e[:-lag]-e[:-lag].mean(); b=e[lag:]-e[lag:].mean()
                va=np.dot(a,a); vbb=np.dot(b,b)
                values.append(float(np.dot(a,b)/np.sqrt(va*vbb)) if min(va,vbb)>1e-12 else 0.)
            peaks.append(max(values))
        u.append(np.mean(peaks))
    cp=power(x[1]); cb=band(True)
    cr=[float(cp[m].sum()/max(cp.sum(),np.finfo(float).tiny)) for m in cb]
    bg=(grid>=1)&(grid<=20)&~cb.any(0)
    for harmonic in (4,8,12,16,20):
        bg&=np.abs(grid-harmonic)>=.5
    q=[float(np.log1p(cp[m].mean()/max(cp[bg].mean()+cp[1:-1].mean()*1e-12,np.finfo(float).tiny))) for m in cb]
    return np.array([r+u,cr+q])


