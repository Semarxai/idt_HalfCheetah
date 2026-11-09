import numpy as np
LN2=np.log(2)
def counts(x): return np.unique(x,return_counts=True)[1]
def H_plugin(c): n=c.sum(); p=c/n; return -(p*np.log2(p)).sum()
def H_mm(c): return H_plugin(c)+(len(c)-1)/(2*c.sum()*LN2)
def H_cs(c):
    n=c.sum(); f1=(c==1).sum()
    if f1==n: f1=n-1
    C=1-f1/n; pa=C*c/n
    return -np.sum(pa*np.log2(pa)/(1-(1-pa)**n))
EST={"plugin":H_plugin,"mm":H_mm,"cs":H_cs}
def window_quantities(S,A,Sn,nS,nA,est):
    f=EST[est]
    sa=S*nA+A; sas=sa*nS+Sn; ssn=S*nS+Sn
    HS,HA,HSn,HSA,HT,HSSn=(f(counts(x)) for x in (S,A,Sn,sa,sas,ssn))
    I=HSA+HSn-HT; IS=HS+HSn-HSSn
    return dict(HS=HS,HA=HA,HSn=HSn,HSA=HSA,HT=HT,I=I,P=I/(HS+HA+HSn),Hf=HT-HSA,Hb=HT-HSn,
                dH=(HT-HSA)-(HT-HSn),I_SSn=IS,I_ASn_S=I-IS,HA_S=HSA-HS)
def coverage(x):
    c=counts(x); n=c.sum(); return len(c)/n, 1-(c==1).sum()/n
