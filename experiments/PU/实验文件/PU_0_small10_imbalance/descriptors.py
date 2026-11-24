"""Vibration envelope R/U and current raw-sideband R/Q in the order domain."""
import torch
from common import LENGTH,SPR,ORDERS

DESCRIPTOR_NAMES=(("R_O","R_I","U_O","U_I"),("Rsb_O","Rsb_I","Qsb_O","Qsb_I"))


def _power(x):
    n=x.shape[-1]
    taper=torch.hann_window(n,periodic=True,dtype=x.dtype,device=x.device)
    x=x-(x*taper).sum(-1,keepdim=True)/taper.sum()
    power=torch.fft.rfft(x*taper).abs().square()
    power[...,0]=power[...,-1]=0
    return power


def _bands(device,dtype,current=False,orders=ORDERS):
    order=torch.fft.rfftfreq(LENGTH,1/SPR,device=device,dtype=dtype)
    centers=torch.tensor(orders,device=device,dtype=dtype)[:,None]*torch.arange(1,3,device=device,dtype=dtype)
    if current:
        centers=centers+4
    width=torch.maximum(.02*centers,centers.new_tensor(2*SPR/LENGTH))
    per_harmonic=(order>=(centers-width)[...,None])&(order<(centers+width)[...,None])
    return per_harmonic.any(1),per_harmonic


def _u(envelope,orders=ORDERS):
    n=envelope.shape[-1]
    e=envelope-envelope.mean(-1,keepdim=True)
    ac=torch.fft.irfft(torch.fft.rfft(e,n=2*n).abs().square(),n=2*n)[...,:n]
    prefix=torch.nn.functional.pad(e.cumsum(-1),(1,0))
    squares=torch.nn.functional.pad(e.square().cumsum(-1),(1,0))
    lag=torch.arange(n,device=e.device)
    count=n-lag
    sa=prefix[...,n-lag]
    sb=prefix[...,-1:]-prefix[...,lag]
    va=(squares[...,n-lag]-sa.square()/count).clamp_min(0)
    vb=(squares[...,-1:]-squares[...,lag]-sb.square()/count).clamp_min(0)
    rho=(ac-sa*sb/count)/(va*vb).sqrt().clamp_min(torch.finfo(e.dtype).tiny)
    rho=torch.where((va>1e-12)&(vb>1e-12),rho.clamp(-1,1),0.)
    result=[]
    for base in orders:
        peaks=[]
        for period in (1,2,3):
            center=SPR*period/base
            width=max(1.,.02*center)
            ids=torch.arange(int(center-width)+1,int(center+width)+1,device=e.device)
            ids=ids[(ids-center).abs()<=width]
            if not len(ids) or int(ids.max())>=n//2:
                raise ValueError("Insufficient actual overlap for the requested period")
            peaks.append(rho[...,ids].amax(-1))
        result.append(torch.stack(peaks).mean(0))
    return torch.stack(result,-1)


@torch.no_grad()
def describe(raw,orders=ORDERS):
    if raw.ndim!=3 or raw.shape[1:]!=(2,LENGTH) or not raw.is_floating_point() or not torch.isfinite(raw).all():
        raise ValueError("Expected finite [batch,2,2560] order-domain inputs")
    dtype=raw.dtype
    x=raw.double()
    x=x-x.mean(-1,keepdim=True)
    rms=x.square().mean(-1,keepdim=True).sqrt()
    active=rms[...,0]>0
    x=x/rms.clamp_min(torch.finfo(x.dtype).tiny)
    n=LENGTH
    hilbert=x.new_zeros(n)
    hilbert[0]=hilbert[n//2]=1
    hilbert[1:n//2]=2
    envelope=torch.fft.ifft(torch.fft.fft(x[:,0])*hilbert).abs()
    vpower=_power(envelope.square())
    vb,_=_bands(x.device,x.dtype,False,orders)
    vr=(vpower[:,None]*vb[None]).sum(-1)/vpower.sum(-1,keepdim=True).clamp_min(torch.finfo(x.dtype).tiny)
    vu=_u(envelope,orders)
    variable=envelope.var(-1,unbiased=False)>1e-12
    vd=torch.cat([vr,vu],-1)
    vd=torch.where((variable&active[:,0])[:,None],vd,0.)
    cpower=_power(x[:,1])
    cb,_=_bands(x.device,x.dtype,True,orders)
    cr=(cpower[:,None]*cb[None]).sum(-1)/cpower.sum(-1,keepdim=True).clamp_min(torch.finfo(x.dtype).tiny)
    order=torch.fft.rfftfreq(n,1/SPR,device=x.device,dtype=x.dtype)
    background=(order>=1)&(order<=20)&~cb.any(0)
    electrical=(order[None]-4*torch.arange(1,6,device=x.device,dtype=x.dtype)[:,None]).abs()<.5
    background&=~electrical.any(0)
    if int(background.sum())<8 or not cb.any(-1).all():
        raise ValueError("Empty or insufficient current signal/background bands")
    peak_density=(cpower[:,None]*cb[None]).sum(-1)/cb.sum(-1)[None]
    background_density=cpower[:,background].mean(-1,keepdim=True)
    floor=cpower[:,1:-1].mean(-1,keepdim=True)*1e-12
    cq=torch.log1p(peak_density/(background_density+floor).clamp_min(torch.finfo(x.dtype).tiny))
    cd=torch.cat([cr,cq],-1)
    cd=torch.where(active[:,1,None],cd,0.)
    result=torch.stack([vd,cd],1).to(dtype)
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite modality-specific descriptors")
    return result


@torch.no_grad()
def statistics(raw):
    x=raw.double(); x=x-x.mean(-1,keepdim=True)
    power=x.square().mean(-1)
    rms=power.sqrt(); active=rms>0
    crest=x.abs().amax(-1)/rms.clamp_min(torch.finfo(x.dtype).tiny)
    kurt=x.pow(4).mean(-1)/power.square().clamp_min(torch.finfo(x.dtype).tiny)
    p=_power(x); p=p/p.sum(-1,keepdim=True).clamp_min(torch.finfo(x.dtype).tiny)
    entropy=-(p*p.clamp_min(torch.finfo(x.dtype).tiny).log()).sum(-1)/x.new_tensor(LENGTH//2-1).log()
    d=torch.stack([rms,crest,kurt,entropy],-1)
    return torch.where(active[...,None],d,0.).to(raw.dtype)
