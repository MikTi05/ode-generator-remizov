"""Optional serial Numba kernels for validated all_methods_v1 tapes.

Not imported by the NumPy backend. No RNG, interpolation, parallel reduction,
fast-math or model-specific compilation. Scratch space is reused per block.
"""
import math
import numpy as np
from numba import njit

JIT=dict(cache=False,fastmath=False,nogil=True,error_model='numpy')


@njit(**JIT)
def _value(tape,x,v):
    for i in range(len(tape)):
        op=int(tape[i,0]);l=int(tape[i,1]);r=int(tape[i,2]);p=tape[i,3];q=tape[i,4]
        if op==0:z=p
        elif op==1:z=math.sin(p*x+q)
        elif op==2:z=math.cos(p*x+q)
        elif op==3:z=math.tanh(p*x+q)
        elif op==4:
            delta=x-q;z=math.exp(-p*delta*delta)
        elif op==5:
            delta=x-q;z=1./(1.+p*delta*delta)
        elif op==6:
            delta=x-p;z=delta/math.sqrt(1.+delta*delta)
        elif op==7:z=math.sin(v[l])
        elif op==8:z=math.cos(v[l])
        elif op==9:z=math.tanh(v[l])
        elif op==10:z=math.exp(-(v[l]*v[l]))
        elif op==11:z=v[l]/(1.+v[l]*v[l])
        elif op==12:z=p*v[l]+q*v[r]
        else:z=v[l]*v[r]
        if not math.isfinite(z):return False
        v[i]=z
    return True


@njit(**JIT)
def _jet(tape,x,v,d,dd):
    for i in range(len(tape)):
        op=int(tape[i,0]);l=int(tape[i,1]);r=int(tape[i,2]);p=tape[i,3];q=tape[i,4]
        if op==0:z=p;dz=0.;ddz=0.
        elif op in (1,2,3):
            theta=p*x+q
            if op==1:z=math.sin(theta);dz=p*math.cos(theta);ddz=-(p*p)*z
            elif op==2:z=math.cos(theta);dz=-p*math.sin(theta);ddz=-(p*p)*z
            else:
                z=math.tanh(theta);s=1.-z*z;dz=p*s;ddz=-2.*p*p*z*s
        elif op==4:
            delta=x-q;z=math.exp(-p*delta*delta);dz=-2.*p*delta*z
            ddz=(4.*p*p*delta*delta-2.*p)*z
        elif op==5:
            delta=x-q;den=1.+p*delta*delta;z=1./den;dz=-2.*p*delta/(den*den)
            ddz=2.*p*(3.*p*delta*delta-1.)/(den*den*den)
        elif op==6:
            delta=x-p;den=1.+delta*delta;z=delta/math.sqrt(den)
            dz=den**-1.5;ddz=-3.*delta*den**-2.5
        elif op==12:
            z=p*v[l]+q*v[r];dz=p*d[l]+q*d[r];ddz=p*dd[l]+q*dd[r]
        elif op==13:
            z=v[l]*v[r];dz=d[l]*v[r]+v[l]*d[r]
            ddz=dd[l]*v[r]+2.*d[l]*d[r]+v[l]*dd[r]
        else:
            u=v[l]
            if op==7:z=math.sin(u);o1=math.cos(u);o2=-z
            elif op==8:z=math.cos(u);o1=-math.sin(u);o2=-z
            elif op==9:z=math.tanh(u);o1=1.-z*z;o2=-2.*z*o1
            elif op==10:z=math.exp(-(u*u));o1=-2.*u*z;o2=(4.*u*u-2.)*z
            else:
                den=1.+u*u;z=u/den;o1=(1.-u*u)/(den*den)
                o2=2.*u*(u*u-3.)/(den*den*den)
            dz=o1*d[l];ddz=o2*d[l]*d[l]+o1*dd[l]
        if not (math.isfinite(z) and math.isfinite(dz) and math.isfinite(ddz)):return False
        v[i]=z;d[i]=dz;dd[i]=ddz
    return True


@njit(**JIT)
def _fused(x,kind,coeff,roots,ytape,params,v,yv,yd,ydd):
    if kind==1:return 1.,0.,0.,1.
    if kind==2:return 1.,0.,0.,(4.-4.*x*x)*math.exp(-x*x)
    if not _value(coeff,x,v):return math.nan,math.nan,math.nan,math.nan
    a=params[0]+params[1]*math.tanh(v[roots[0]])
    b=params[2]*math.tanh(v[roots[1]])
    c=params[3]*math.tanh(v[roots[2]])
    if not _jet(ytape,x,yv,yd,ydd):return math.nan,math.nan,math.nan,math.nan
    z=(x-params[6])/params[7];phi=0.;bp=0.;bpp=0.
    if abs(z)<1.:
        q=1.-z*z;phi=math.exp(1.-1./q)
        if phi!=0.:
            dl=-2.*z/(q*q);ddl=-2./(q*q)-8.*z*z/(q*q*q)
            bp=phi*dl/params[7]
            bpp=phi*(ddl+dl*dl)/(params[7]*params[7])
    k=len(ytape)-1;s=params[5]
    y=s*phi*yv[k]
    dy=s*(bp*yv[k]+phi*yd[k])
    d2y=s*(bpp*yv[k]+2.*bp*yd[k]+phi*ydd[k])
    g=-(a*d2y+b*dy+(c-params[4])*y)
    return a,b,c,g


@njit(**JIT)
def evaluate_batch(x,kind,coeff,roots,ytape,params):
    output=np.empty((4,len(x)))
    v=np.empty(len(coeff));yv=np.empty(len(ytape));yd=np.empty(len(ytape));ydd=np.empty(len(ytape))
    for j in range(len(x)):
        a,b,c,g=_fused(x[j],kind,coeff,roots,ytape,params,v,yv,yd,ydd)
        output[0,j]=a;output[1,j]=b;output[2,j]=c;output[3,j]=g
    return output


@njit(**JIT)
def advance_block(x,a,b,c,total,logw,normals,h,offset,N,kind,coeff,roots,ytape,params,bounds):
    """Same EM/trapezoids, time-major normal order; return status + actual updates.

    Status: 0 ok,1 nonpositive diffusion,2 bound violation,3 nonfinite state,
    4 coefficient evaluation failure,5 numerical overflow. No shared RNG.
    """
    v=np.empty(len(coeff));yv=np.empty(len(ytape));yd=np.empty(len(ytape));ydd=np.empty(len(ytape))
    sh=math.sqrt(h);s2=math.sqrt(2.);updates=0
    for k in range(len(normals)):
        for j in range(len(x)):
            if not math.isfinite(a[j]) or not math.isfinite(b[j]):return 4,updates
            if a[j]<=0:return 1,updates
            if a[j]<bounds[0] or a[j]>bounds[1] or abs(b[j])>bounds[2]:return 2,updates
            newx=x[j]+b[j]*h+((math.sqrt(a[j])*sh)*s2)*normals[k,j]
            updates+=1
            if not math.isfinite(newx):return 3,updates
            an,bn,cn,gn=_fused(newx,kind,coeff,roots,ytape,params,v,yv,yd,ydd)
            if not (math.isfinite(an) and math.isfinite(bn) and math.isfinite(cn) and math.isfinite(gn)):
                return 4,updates
            if abs(cn)>bounds[3] or abs(gn)>bounds[5]:return 2,updates
            newlog=logw[j]+h*(.5*(c[j]-bounds[4])+.5*(cn-bounds[4]))
            if not math.isfinite(newlog):return 5,updates
            weight=.5*h if offset+k+1==N else h
            value=total[j]+weight*math.exp(newlog)*gn
            if not math.isfinite(value):return 3,updates
            x[j]=newx;a[j]=an;b[j]=bn;c[j]=cn;logw[j]=newlog;total[j]=value
    return 0,updates
