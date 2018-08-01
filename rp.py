#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 19:52:58 2018

@author: iainanderson
"""

import numpy as np
from scipy.integrate import quad
import scipy as spy
from scipy.integrate import odeint
import math

def Anis_Backus(f,c1,c2,c3):
    c11=np.zeros(3)
    c33=np.zeros(3)
    c13=np.zeros(3)
    c44=np.zeros(3)
    c66=np.zeros(3)
    
    
    c11[0] = c1[0,0]
    c33[0] = c1[2,2]
    c13[0] = c1[0,2]
    c44[0] = c1[3,3]
    c66[0] = c1[5,5]
    
    c11[1] = c2[0,0]
    c33[1] = c2[2,2]
    c13[1] = c2[0,2]
    c44[1] = c2[3,3]
    c66[1] = c2[5,5]
    
    c11[2] = c3[0,0]
    c33[2] = c3[2,2]
    c13[2] = c3[0,2]
    c44[2] = c3[3,3]
    c66[2] = c3[5,5]
    
    c11_eff = sum(f*(c11-(c13**2)*(1/c33))) + (1/sum((f*(1/c33))))*(sum(f*(1/c33)*c13))**2
    c33_eff = 1/(sum(f*(1/c33)))
    c13_eff = (1/(sum(f*(1./c33))))*sum((f*(c13/c33)))
    c44_eff = 1/(sum(f*(1/c44)))
    c66_eff = sum(f*c66)
    c12_eff=c11_eff-(2*c66_eff)
    
    cout=np.array([[c11_eff,c12_eff,c13_eff,0,      0,      0],
                   [c12_eff,c11_eff,c13_eff,0,      0,      0],
                   [c13_eff,c13_eff,c33_eff,0,      0,      0],
                   [0,      0,      0,      c44_eff,0,      0],
                   [0,      0,      0,      0,      c44_eff,0],
                   [0,      0,      0,      0,      0,      c66_eff]])
    
    return (cout)

def HS(f,k,u):

    c=4/3
    
    kmx=max(k)
    kmn=min(k)
    umx=max(u)
    umn=min(u)
    
    #HS upper bound
    k_u=1/sum(f/(k+c*umx))-c*umx
    # HS lower bound	
    k_l=1/sum(f/(k+c*umn))-c*umn	
    
    etamx=umx*(9*kmx+8*umx)/(kmx+2*umx)/6
    etamn=umn*(9*kmn+8*umn)/(kmn+2*umn)/6
    
    # HS upper bound
    u_u=1/sum(f/(u+etamx))-etamx
    # HS lower bound
    u_l=1/sum(f/(u+etamn))-etamn	
    
    ka=(k_u+k_l)/2			
    ua=(u_u+u_l)/2
    
    return ka,ua


def VRH(k,u,f):
    
    k_u=(f*k).sum(axis=0)
    k_l=1/(f/k).sum(axis=0)	
		

    u_u=(f*u).sum(axis=0)
    u_l=1/(f/u).sum(axis=0)	

    ka=(k_u+k_l)/2			
    ua=(u_u+u_l)/2
    
    return ka,ua

def hudson1(ec,ar,Kfl,rhofl,K,G,rho,ax):
    
    pi=np.pi
    
    lam=K-2/3*G
    mu=G
    kapa=Kfl*(lam+2*mu)/(pi*ar*mu*(lam+mu))
    u3=4/3*(lam+2*mu)/((lam+mu)*(1+kapa))
    u1=16/3*(lam+2*mu)/(3*lam+4*mu)
    c11=lam+2*mu-lam**2*ec*u3/mu
    c13=lam-lam*(lam+2*mu)*ec*u3/mu
    c33=lam+2*mu-(lam+2*mu)**2*ec*u3/mu
    c44=mu-mu*ec*u1
    c66=mu
    
    Ctih=np.zeros((6,6))
    # ax=1 == HTI, ax=3==VTI
    if ax==1:
        Ctih[0,0]=c33
        Ctih[1,1]=c11
        Ctih[2,2]=c11
        Ctih[0,2]=c13
        Ctih[2,0]=c13
        Ctih[0,1]=c13
        Ctih[1,0]=c13
        Ctih[1,2]=c11-2*c66
        Ctih[2,1]=Ctih[1,2]
        Ctih[3,3]=c66
        Ctih[4,4]=c44
        Ctih[5,5]=c44
    
    elif ax==3:
        Ctih[0,0]=c11
        Ctih[1,1]=c11
        Ctih[2,2]=c33;
        Ctih[0,2]=c13;
        Ctih[2,0]=c13;
        Ctih[0,1]=c11-2*c66
        Ctih[1,0]=Ctih[0,1]
        Ctih[1,2]=c13
        Ctih[2,1]=Ctih[1,2]
        Ctih[3,3]=c44
        Ctih[4,4]=c44
        Ctih[5,5]=c66
    
    
    
    phi=(4*pi/3)*ar*ec
    den=(1-phi)*rho + phi*rhofl
    return Ctih,den



def CSiso(k,u):
    if u==0:
        u=u+0.000001
    lmda = k - 2*u/3;
    c11 = lmda + 2*u;
    c12 = lmda;
    c44 = u;
    
    c = np.zeros([6,6])
    c[0,0] = c11;
    c[0,1] = c12;
    c[0,2] = c12;
    c[1,0] = c12;
    c[1,1] = c11;
    c[1,2] = c12;
    c[2,0] = c12;
    c[2,1] = c12;
    c[2,2] = c11;
    c[3,3] = c44;
    c[4,4] = c44;
    c[5,5] = c44;
    return c

def fint(x,cb11,cb12,cb13,cb33,cb44,asp,flag):
    
    d = cb11;
    e = (cb11 - cb12)/2;
    f = cb44;
    g = cb13 + cb44;
    h = cb33;
    r = float(1/asp);
    delta=1/((np.dot(e,(1-x**2))+np.dot((np.dot(f,r**2)),x**2))*((np.dot(d,1-x**2)+np.dot(np.dot(f,r**2),x**2))*(np.dot(f,1-x**2)+np.dot(np.dot(h,r**2),x**2))-(np.dot(np.dot(g**2,r**2),x**2)*(1-x**2))))
    
    case={
        1:np.dot(np.pi/2,delta)*(1-x**2)*((np.dot(f,1-x**2)+np.dot(np.dot(h,r**2),x**2))*((np.dot((np.dot(3,e)+d),1-x**2)+np.dot(np.dot(np.dot(4,f),r**2),x**2)))-np.dot(np.dot(g**2,r**2),x**2)*(1-x**2)),
        
    
        2:np.dot(np.dot(np.dot(4,np.pi),r**2),delta)*x**2*(np.dot(d,1-x**2)+np.dot(np.dot(f,r**2),x**2))*(np.dot(e,1-x**2)+np.dot(np.dot(f,r**2),x**2)),
                 
   
        3:np.dot(np.pi/2,delta)*(1-x**2)*((np.dot(f,1-x**2)+np.dot(np.dot(h,r**2),x**2))*(np.dot((e+np.dot(3,d)),1-x**2)+np.dot(np.dot(np.dot(4,f),r**2),x**2))-np.dot(np.dot(np.dot(3,g**2),r**2),x**2)*(1-x**2)),
        
    
        4:np.dot(np.dot(np.dot(2,np.pi),r**2),delta)*x**2*((np.dot(d+e,1-x**2)+np.dot(np.dot(np.dot(2,f),r**2),x**2))*(np.dot(f,1-x**2)+np.dot(np.dot(h,r**2),x**2))-np.dot(np.dot(np.dot(g**2,r**2),x**2),1-x**2)),
                        
    
        5:np.dot(np.dot(2,np.pi),delta)*(1-x**2)*(np.dot(d,1-x**2)+np.dot(np.dot(f,r**2),x**2))*(np.dot(e,1-x**2)+np.dot(np.dot(f,r**2),x**2)),
        
    
        6:np.dot(np.pi/2,delta)*((1-x**2)**2)*(np.dot(np.dot(g**2,r**2),x**2)-np.dot(d-e,(np.dot(f,1-x**2)+np.dot(np.dot(h,r**2),x**2)))),
        
    
        7:np.dot(np.dot(np.dot(np.dot(-2,np.pi),g),r**2),delta)*x**2*(1-x**2)*(np.dot(e,1-x**2)+np.dot(np.dot(f,r**2),x**2)),
    }
    return(case.get(flag));


def Anis_SCA(c1,c2,x1,x2,asp1,asp2):
    
    
    # Inclusion Phases
    k1=c1[0,0]-4/3*c1[3,3]
    mu1=c1[3,3];
    
    k2=c2[0,0]-4/3*c2[3,3];
    mu2=c2[3,3];
    
    # reshape c1 matrix
    a=c1[:,3].reshape(6,1)
    b=c1[:,4].reshape(6,1)
    c=c1[:,5].reshape(6,1)
    x=np.hstack((c1,a,b,c))
    d=x[3,:]
    e=x[4,:]
    f=x[5,:]
    c1=np.vstack((x,d,e,f))
    
    # reshape c2 matrix
    a=c2[:,3].reshape(6,1)
    b=c2[:,4].reshape(6,1)
    c=c2[:,5].reshape(6,1)
    x=np.hstack((c2,a,b,c))
    d=x[3,:]
    e=x[4,:]
    f=x[5,:]
    c2=np.vstack((x,d,e,f))
    
    
    
    # Voigt average for inital sca model
    ksca=(k1*x1)+k2*(1-x1);
    musca=(mu1*x1)+(mu2*(1-x1));
    Csca=CSiso(ksca,musca)
    
    # Reshape Csca
    a=Csca[:,3].reshape(6,1)
    b=Csca[:,4].reshape(6,1)
    c=Csca[:,5].reshape(6,1)
    x=np.hstack((Csca,a,b,c))
    d=x[3,:]
    e=x[4,:]
    f=x[5,:]
    Csca=np.vstack((x,d,e,f))
    
    
    knew=0
    I=np.identity(9)
    tol=1e-6*Csca[0,0]
    del_var=abs(ksca-knew)
    niter=0
    
    
    # iteration loop for Csa
    
    while ((del_var > abs(tol)) and (niter<3000)):
         
    
        c11=Csca[0,0]
        c12=c11-2*Csca[5,5]
        c13=Csca[0,2]
        c33=Csca[2,2]
        c44=Csca[3,3]
                  
        
        f1=quad((lambda x:fint(x,c11,c12,c13,c33,c44,asp1,1)),0,1)
        f2=quad((lambda x:fint(x,c11,c12,c13,c33,c44,asp1,2)),0,1)
        f3=quad((lambda x:fint(x,c11,c12,c13,c33,c44,asp1,3)),0,1)
        f4=quad((lambda x:fint(x,c11,c12,c13,c33,c44,asp1,4)),0,1)
        f5=quad((lambda x:fint(x,c11,c12,c13,c33,c44,asp1,5)),0,1)
        f6=quad((lambda x:fint(x,c11,c12,c13,c33,c44,asp1,6)),0,1)
        f7=quad((lambda x:fint(x,c11,c12,c13,c33,c44,asp1,7)),0,1)
        
        g1111=f1[0];
        g2222=f1[0];
        g3333=f2[0];
        g1122=f3[0];g2211=f3[0];
        g1133=f4[0];g2233=f4[0];
        g3311=f5[0];g3322=f5[0];
        g1212=f6[0];
        g1313=f7[0];g2323=f7[0];
        
        G1111=(g1111+g1111)/(8*np.pi);
        G1122=(g1212+g1212)/(8*np.pi);   
        G1133=(g1313+g1313)/(8*np.pi);
        G2211=(g1212+g1212)/(8*np.pi);
        G2222=(g2222+g2222)/(8*np.pi);
        G2233=(g2323+g2323)/(8*np.pi);
        G3311=(g1313+g1313)/(8*np.pi); 
        G3322=(g2323+g2323)/(8*np.pi);
        G3333=(g3333+g3333)/(8*np.pi);
        G2323=(g2233+g2323)/(8*np.pi);
        G1313=(g1133+g1313)/(8*np.pi);
        G1212=(g1122+g1212)/(8*np.pi);
        G2332=(g2323+g3322)/(8*np.pi);
        G1331=(g1313+g3311)/(8*np.pi);
        G1221=(g1212+g2211)/(8*np.pi);
        G3223=(g2323+g2233)/(8*np.pi);
        G3113=(g1313+g1133)/(8*np.pi);
        G2112=(g1212+g1122)/(8*np.pi);
        G3232=(g3322+g2323)/(8*np.pi);
        G3131=(g3311+g1313)/(8*np.pi);
        G2121=(g2211+g1212)/(8*np.pi);
        
        G1=np.array([[G1111,G1122,G1133,0., 0., 0., 0., 0., 0.],
                    [G2211,G2222,G2233,0., 0., 0., 0., 0., 0.],
                    [G3311,G3322,G3333,0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., G2323,0., 0., G2332,0., 0.],
                    [0., 0., 0., 0., G1313,0., 0., G1331,0.],
                    [0., 0., 0., 0., 0., G1212,0., 0., G1221],
                    [0., 0., 0., G3223,0., 0., G3232,0., 0.],
                    [0., 0., 0., 0., G3113,0., 0., G3131,0.],
                    [0., 0., 0., 0., 0., G2112,0., 0., G2121]])
        
        
        #-------------------------------------------------------------------------
            
        f1=quad((lambda x:fint(x,c11,c12,c13,c33,c44,asp2,1)),0,1)
        f2=quad((lambda x:fint(x,c11,c12,c13,c33,c44,asp2,2)),0,1)
        f3=quad((lambda x:fint(x,c11,c12,c13,c33,c44,asp2,3)),0,1)
        f4=quad((lambda x:fint(x,c11,c12,c13,c33,c44,asp2,4)),0,1)
        f5=quad((lambda x:fint(x,c11,c12,c13,c33,c44,asp2,5)),0,1)
        f6=quad((lambda x:fint(x,c11,c12,c13,c33,c44,asp2,6)),0,1)
        f7=quad((lambda x:fint(x,c11,c12,c13,c33,c44,asp2,7)),0,1)
            
        g1111=f1[0];
        g2222=f1[0];
        g3333=f2[0];
        g1122=f3[0];g2211=f3[0];
        g1133=f4[0];g2233=f4[0];
        g3311=f5[0];g3322=f5[0];
        g1212=f6[0];
        g1313=f7[0];g2323=f7[0];
        
        G1111=(g1111+g1111)/(8*np.pi);
        G1122=(g1212+g1212)/(8*np.pi);   
        G1133=(g1313+g1313)/(8*np.pi);
        G2211=(g1212+g1212)/(8*np.pi);
        G2222=(g2222+g2222)/(8*np.pi);
        G2233=(g2323+g2323)/(8*np.pi);
        G3311=(g1313+g1313)/(8*np.pi); 
        G3322=(g2323+g2323)/(8*np.pi);
        G3333=(g3333+g3333)/(8*np.pi);
        G2323=(g2233+g2323)/(8*np.pi);
        G1313=(g1133+g1313)/(8*np.pi);
        G1212=(g1122+g1212)/(8*np.pi);
        G2332=(g2323+g3322)/(8*np.pi);
        G1331=(g1313+g3311)/(8*np.pi);
        G1221=(g1212+g2211)/(8*np.pi);
        G3223=(g2323+g2233)/(8*np.pi);
        G3113=(g1313+g1133)/(8*np.pi);
        G2112=(g1212+g1122)/(8*np.pi);
        G3232=(g3322+g2323)/(8*np.pi);
        G3131=(g3311+g1313)/(8*np.pi);
        G2121=(g2211+g1212)/(8*np.pi);
        
        G2=np.array([[G1111,G1122,G1133,0., 0., 0., 0., 0., 0.],
                    [G2211,G2222,G2233,0., 0., 0., 0., 0., 0.],
                    [G3311,G3322,G3333,0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., G2323,0., 0., G2332,0., 0.],
                    [0., 0., 0., 0., G1313,0., 0., G1331,0.],
                    [0., 0., 0., 0., 0., G1212,0., 0., G1221],
                    [0., 0., 0., G3223,0., 0., G3232,0., 0.],
                    [0., 0., 0., 0., G3113,0., 0., G3131,0.],
                    [0., 0., 0., 0., 0., G2112,0., 0., G2121]])
        
        
        Q1=spy.linalg.inv(I+np.dot(G1,c1-Csca));
        Q2=spy.linalg.inv(I+np.dot(G2,c2-Csca));
        
        Cnew=np.dot((np.dot((x1*c1),Q1)+np.dot((x2*c2),Q2)),spy.linalg.inv(x1*Q1+x2*Q2)); 
        knew=Cnew[0,0]-4/3*Cnew[3,3]
        del_var=abs(ksca-knew)
            
        Csca=Cnew
        ksca=knew
               
        niter=niter+1
        
    return Csca

def demyprime(y,t,Ci,asp):
    
    # Reshape Csca
    a=Ci[:,3].reshape(6,1)
    b=Ci[:,4].reshape(6,1)
    c=Ci[:,5].reshape(6,1)
    x=np.hstack((Ci,a,b,c))
    d=x[3,:]
    e=x[4,:]
    f=x[5,:]
    Ci=np.vstack((x,d,e,f))
    
    cb11=y[0]; cb33=y[1]; cb13=y[2]; cb44=y[3]; cb66=y[4];
    cb12=cb11-(2*cb66)
    
    Cb=np.array([[cb11,cb12,cb13,0., 0., 0., 0., 0., 0.],
                 [cb12,cb11,cb13,0., 0., 0., 0., 0., 0.],
                 [cb13,cb13,cb33,0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., cb44,0., 0., cb44,0., 0.],
                 [0., 0., 0., 0., cb44,0., 0., cb44,0.],
                 [0., 0., 0., 0., 0., cb66,0., 0., cb66],
                 [0., 0., 0., cb44,0., 0., cb44,0., 0.],
                 [0., 0., 0., 0., cb44,0., 0., cb44,0.],
                 [0., 0., 0., 0., 0., cb66,0., 0., cb66]])
    
    
    yprime=spy.zeros(5);
    I=np.identity(9)
    
    
    
    #f1=quad((lambda x:fint(x,cb11,cb12,cb13,cb33,cb44,asp,1)),0,1)
    f1=quad((lambda x:fint(x,cb11,cb12,cb13,cb33,cb44,asp,1)),0,1,epsabs=1e-13,epsrel=1e-8)
    f2=quad((lambda x:fint(x,cb11,cb12,cb13,cb33,cb44,asp,2)),0,1,epsabs=1e-13,epsrel=1e-8)
    f3=quad((lambda x:fint(x,cb11,cb12,cb13,cb33,cb44,asp,3)),0,1,epsabs=1e-13,epsrel=1e-8)
    f4=quad((lambda x:fint(x,cb11,cb12,cb13,cb33,cb44,asp,4)),0,1,epsabs=1e-13,epsrel=1e-8)
    f5=quad((lambda x:fint(x,cb11,cb12,cb13,cb33,cb44,asp,5)),0,1,epsabs=1e-13,epsrel=1e-8)
    f6=quad((lambda x:fint(x,cb11,cb12,cb13,cb33,cb44,asp,6)),0,1,epsabs=1e-13,epsrel=1e-8)
    f7=quad((lambda x:fint(x,cb11,cb12,cb13,cb33,cb44,asp,7)),0,1,epsabs=1e-13,epsrel=1e-8)
    
    g1111=f1[0];
    g2222=f1[0];
    g3333=f2[0];
    g1122=f3[0];g2211=f3[0];
    g1133=f4[0];g2233=f4[0];
    g3311=f5[0];g3322=f5[0];
    g1212=f6[0];
    g1313=f7[0];g2323=f7[0];

    #Gijkl=1/(8*pi)*(gikjl+gjkil)
    G1111=(g1111+g1111)/(8*np.pi);
    G1122=(g1212+g1212)/(8*np.pi);   
    G1133=(g1313+g1313)/(8*np.pi);
    G2211=(g1212+g1212)/(8*np.pi);
    G2222=(g2222+g2222)/(8*np.pi);
    G2233=(g2323+g2323)/(8*np.pi);
    G3311=(g1313+g1313)/(8*np.pi); 
    G3322=(g2323+g2323)/(8*np.pi);
    G3333=(g3333+g3333)/(8*np.pi);
    G2323=(g2233+g2323)/(8*np.pi);
    G1313=(g1133+g1313)/(8*np.pi);
    G1212=(g1122+g1212)/(8*np.pi);
    G2332=(g2323+g3322)/(8*np.pi);
    G1331=(g1313+g3311)/(8*np.pi);
    G1221=(g1212+g2211)/(8*np.pi);
    G3223=(g2323+g2233)/(8*np.pi);
    G3113=(g1313+g1133)/(8*np.pi);
    G2112=(g1212+g1122)/(8*np.pi);
    G3232=(g3322+g2323)/(8*np.pi);
    G3131=(g3311+g1313)/(8*np.pi);
    G2121=(g2211+g1212)/(8*np.pi);

    G=np.array([[G1111,G1122,G1133,0., 0., 0., 0., 0., 0.],
                [G2211,G2222,G2233,0., 0., 0., 0., 0., 0.],
                [G3311,G3322,G3333,0., 0., 0., 0., 0., 0.],
                [0., 0., 0., G2323,0., 0., G2332,0., 0.],
                [0., 0., 0., 0., G1313,0., 0., G1331,0.],
                [0., 0., 0., 0., 0., G1212,0., 0., G1221],
                [0., 0., 0., G3223,0., 0., G3232,0., 0.],
                [0., 0., 0., 0., G3113,0., 0., G3131,0.],
                [0., 0., 0., 0., 0., G2112,0., 0., G2121]])
    
    Q=I+np.dot(G,Ci-Cb);
    K=np.dot(Ci-Cb,spy.linalg.inv(Q));
      
    yprime[0]=K[0,0]/(1-t)
    yprime[1]=K[2,2]/(1-t)
    yprime[2]=K[0,2]/(1-t)
    yprime[3]=K[3,3]/(1-t)
    yprime[4]=K[5,5]/(1-t)


    return yprime

def Anis_DEM(Cb,Ci,asp):
    
    tspan=np.linspace(0, 1, 100)
    init=np.array([Cb[0,0],Cb[2,2],Cb[0,2],Cb[3,3],Cb[5,5]])
    dem=odeint(demyprime,init,tspan,args=(Ci,asp))
    return dem


def Anis_SCA_DEM(c1,c2,x2,asp1,asp2):

    #Initial SCA model
    [c_sca]=Anis_SCA(c1,c2,0.5,0.5,asp1,asp2);

    #bypass if x2=0.5;
    if x2==0.5:
        cout=c_sca

    #add phase 1 if x2 is less than 0.5
    elif x2 < 0.5: 
        Cb=c_sca;
        Ci=c1;
        asp=asp1;
        xi=((1-x2)-0.5)*2;
        cout=Anis_DEM(Cb,Ci,asp,xi);

    #add phase 1 if x2 is less than 0.5
    elif x2 > 0.5:
        Cb=c_sca;
        Ci=c2;
        asp=asp2;
        xi=(x2-0.5)*2;
        cout=Anis_DEM(Cb,Ci,asp,xi)
        
    return cout

def berryscm(k,mu,asp,x):

    kbr=[] 
    mubr=[]

    k=np.reshape(k,(-1,1)) 
    mu=np.reshape(mu,(-1,1))
    asp=np.reshape(asp,(-1,1))
    x=np.reshape(x,(-1,1))
    indx=np.where(asp==1)[0]
    for i in indx:
        asp[i]=0.99
    
    theta=np.zeros(asp.shape)
    fn=np.zeros(asp.shape)
    
# Oblate spheroids
    obdx=np.where(asp<1)[0]
    for i in obdx:
        theta[i]=(asp[i]/((1-asp[i]**2)**(3/2)))*(math.acos(asp[i])-asp[i]*math.sqrt(1-asp[i]**2))
        fn[i]=(asp[i]**2/(1-asp[i]**2))*(3.*theta[i]-2)
    
# Prolate spheroids
    prdx=np.where(asp>1)[0]
    for i in prdx:
        theta[i]=(asp[i]/((asp[i]**2-1)**(3/2)))*(asp[i]*math.sqrt(asp[i]**2-1)-math.acosh(asp[i]))
        fn[i]=(asp[i]**2/(asp[i]**2-1))*(2-3*theta[i])
    

    ksc= sum(k*x)
    musc= sum(mu*x)
    knew=0
    munew=0
    tol=1e-6*k[0]
    delz=abs(ksc-knew)
    niter=0
    
    while (delz> abs(tol)) & (niter<3000):
        nusc=(3*ksc-2*musc)/(2*(3*ksc+musc))
        a=mu/musc-1
        b=(1/3)*(k/ksc-mu/musc)
        r=(1-2*nusc)/(2*(1-nusc))

        f1=1+a*((3/2)*(fn+theta)-r*((3/2)*fn+(5/2)*theta-(4/3)))
        f2=1+a*(1+(3/2)*(fn+theta)-(r/2)*(3*fn+5*theta))+b*(3-4*r)
        f2=f2+(a/2)*(a+3*b)*(3-4*r)*(fn+theta-r*(fn-theta+2*theta**2))
        f3=1+a*(1-(fn+(3/2)*theta)+r*(fn+theta))
        f4=1+(a/4)*(fn+3*theta-r*(fn-theta))
        f5=a*(-fn+r*(fn+theta-(4/3))) + b*theta*(3-4*r)
        f6=1+a*(1+fn-r*(fn+theta))+b*(1-theta)*(3-4*r)
        f7=2+(a/4)*(3.*fn+9*theta-r*(3.*fn+5*theta)) + b*theta*(3-4*r)
        f8=a*(1-2*r+(fn/2)*(r-1)+(theta/2)*(5*r-3))+b*(1-theta)*(3-4*r)
        f9=a*((r-1)*fn-r*theta) + b*theta*(3-4*r)

        p=3*f1/f2
        q=(2/f3) + (1/f4) +((f4*f5 + f6*f7 - f8*f9)/(f2*f4))
        p=p/3
        q=q/5
        
        
        knew= sum(x*k*p)/sum(x*p)
        munew= sum(x*mu*q)/sum(x*q)
               
        delz=abs(ksc-knew)
        ksc=knew
        musc=munew
        niter=niter+1;
                                
    kbr=ksc
    mubr=musc
    
    return (kbr),(mubr)

def berryscm_crack(k1,u1,k2,u2,asp1,asp2,x1,x2):

    def beta(k,u):
        B = u*(3*k + u)/(3*k + 4*u)
        return B
    
    # Initial guess (VRH average)
    k=np.array([k1,k2])
    u=np.array([u1,u2])
    f=np.array([x1,x2])
    ksc,musc=VRH(k,u,f)
    
    knew=0
    munew=0
    tol=1e-6*ksc
    delz=abs(ksc-knew)
    niter=0
    
    while (delz> abs(tol)) & (niter<3000):
        
        # For inclusions of phase1
        P1 = (ksc + 4/3*u1)/(k1 + 4/3*u1 + np.pi*alpha*beta(ksc, musc))
        Q1 = 1/5*(1 +
                   8*musc / (4*u1 + np.pi*alpha*(musc + 2*beta(ksc, musc))) +
                   2*(k1 + 2/3*(u1 + musc)) /
                   (k1 + 4/3*u1 + np.pi*alpha*beta(ksc, musc)))
  
       # For inclusions of phase2
        P2 = (ksc + 4/3*u2)/(k2 + 4/3*u2 + np.pi*alpha*beta(ksc, musc))
        Q2 = 1/5*(1 +
                   8*musc / (4*u2 + np.pi*alpha*(musc + 2*beta(ksc, musc))) +
                   2*(k2 + 2/3*(u2 + musc)) /
                   (k2 + 4/3*u2 + np.pi*alpha*beta(ksc, musc)))
        
        
        knew=(x1*(k1-ksc)*P1)+(x2*(k2-ksc)*P2)
        munew=(x1*(u1-musc)*Q1)+(x2*(u2-musc)*Q2)
        
        delz=abs(ksc-knew)
        ksc=knew
        musc=munew
        niter=niter+1;
                                
    kbr=ksc
    mubr=musc
    
    return kbr,mubr

