import os, numpy as np, pandas as pd, statsmodels.api as sm

NAMES={1:"Yoplait",2:"Chobani",3:"Dannon",4:"Stonyfield Farm",5:"Activia"}
NESTMAP={1:"sugary",2:"nosugar",3:"nosugar",4:"sugary",5:"sugary"}

def _booktabs_matrix(M,rows,cols,caption,label):
    head="\\begin{table}[H]\n\\centering\n"
    head+=f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
    head+="\\begin{tabular}{l"+"r"*len(cols)+"}\n\\toprule\n"
    head+=" & "+" & ".join(cols)+" \\\\\n\\midrule\n"
    body="\n".join([r+" & "+" & ".join(f"{v:.4f}" for v in M[i,:])+" \\\\" for i,r in enumerate(rows)])
    tail="\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return head+body+tail

def _ehw_2sls(y,X,Z):
    if hasattr(y,"values"): y=y.values
    if hasattr(X,"values"): X=X.values
    if hasattr(Z,"values"): Z=Z.values
    y=y.reshape(-1,1).astype(float); X=X.astype(float); Z=Z.astype(float)
    ZZ=Z.T@Z; ZZinv=np.linalg.inv(ZZ); XZ=X.T@Z; ZY=Z.T@y
    beta=np.linalg.inv(XZ@ZZinv@XZ.T)@(XZ@ZZinv@ZY)
    u=(y-X@beta).reshape(-1)
    S=(Z*(u**2)[:,None]).T@Z
    A=XZ@ZZinv@XZ.T
    B=XZ@ZZinv@S@ZZinv@XZ.T
    cov=np.linalg.inv(A)@B@np.linalg.inv(A)
    n,k=X.shape
    cov=cov*n/(n-k)
    se=np.sqrt(np.diag(cov)).reshape(-1,1)
    return beta.flatten()

def _fes(df):
    d_prod=pd.get_dummies(df["product"],prefix="prod",drop_first=True)
    d_city=pd.get_dummies(df["city"],prefix="city",drop_first=True)
    d_per=pd.get_dummies(df["period"],prefix="per",drop_first=True)
    FE=pd.concat([d_prod,d_city,d_per],axis=1)
    FE=sm.add_constant(FE,has_constant="add")
    return FE

def _prep(df,zdist,zdiesel):
    for c in ["price","share",zdist,zdiesel]:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    df["nest"]=df["product"].map(NESTMAP)
    df["market"]=list(zip(df["city"],df["period"]))
    inside=df.groupby("market")["share"].sum().rename("inside_share_sum")
    df=df.merge(inside,on="market",how="left")
    df["s0"]=1.0-df["inside_share_sum"]
    gsum=df.groupby(["market","nest"])["share"].sum().rename("s_g")
    df=df.merge(gsum,on=["market","nest"],how="left")
    eps=1e-12
    df["s_jg"]=np.clip(df["share"]/np.clip(df["s_g"],eps,None),eps,1-1e-12)
    df["log_s_jg"]=np.log(df["s_jg"])
    df["logit_dep_nl"]=np.log(np.clip(df["share"],eps,None))-np.log(np.clip(df["s0"],eps,None))
    df["iv_cost"]=df[zdist]*df[zdiesel]
    ncount=df.groupby(["market","nest"])["product"].transform("count")
    df["iv_N"]=ncount.astype(float)
    return df

def estimate_alpha_rho(df):
    FE=_fes(df)
    X=pd.concat([df[["price","log_s_jg"]],FE],axis=1)
    Z=pd.concat([df[["iv_cost","iv_N"]],FE],axis=1)  
    X=X.replace([np.inf,-np.inf],np.nan).apply(pd.to_numeric,errors="coerce").astype(float)
    Z=Z.replace([np.inf,-np.inf],np.nan).apply(pd.to_numeric,errors="coerce").astype(float)
    y=pd.to_numeric(df["logit_dep_nl"].replace([np.inf,-np.inf],np.nan),errors="coerce").astype(float)
    ok=y.notna()&X.notna().all(axis=1)&Z.notna().all(axis=1)
    y,X,Z=y.loc[ok],X.loc[ok],Z.loc[ok]
    b=_ehw_2sls(y,X,Z)
    cols=X.columns.tolist()
    a=-float(b[cols.index("price")])
    r=float(b[cols.index("log_s_jg")])
    return a,r

def elasticities_and_diversion(df,a,r,city=1,period=1):
    m=df[(df["city"]==city)&(df["period"]==period)].copy().sort_values("product")
    m["nest"]=m["product"].map(NESTMAP)
    s=m["share"].to_numpy(float); p=m["price"].to_numpy(float)
    prods=m["product"].tolist()
    names=[NAMES.get(int(j),str(int(j))) for j in prods]
    gsum=m.groupby("nest")["share"].transform("sum").to_numpy(float)
    s_jg=np.clip(s/np.clip(gsum,1e-12,None),1e-12,1-1e-12)
    J=len(names)
    E=np.zeros((J+1,J))
    for k in range(J):
        E[0,k]=a*p[k]*s[k]
        for j in range(J):
            if j==k:
                E[j+1,k]=-a*p[k]*(1.0 - r*(1.0 - s_jg[k]) - s[k])
            else:
                same=(m.iloc[j]["nest"]==m.iloc[k]["nest"])
                if same:
                    E[j+1,k]=a*p[k]*(r*s_jg[k]-s[k])
                else:
                    E[j+1,k]=a*p[k]*s[k]
    D=np.zeros((J,J+1))
    for j in range(J):
        d_sj=-a*s[j]*(1.0 - r*(1.0 - s_jg[j]) - s[j])
        for k in range(J):
            if j==k: 
                D[j,k]=0.0
            else:
                same=(m.iloc[j]["nest"]==m.iloc[k]["nest"])
                ds_k = a*s[k]*(r*s_jg[j]-s[j]) if same else a*s[k]*s[j]
                D[j,k]=(-ds_k)/(-d_sj)
        ds0 = -np.sum([a*s[midx]*(r*s_jg[j]-s[j]) if (m.iloc[midx]["nest"]==m.iloc[j]["nest"] and midx!=j) else a*s[midx]*s[j] if midx!=j else d_sj for midx in range(J)])
        D[j,J]=(-ds0)/(-d_sj)
    return E,["Outside Option"]+names,names,D,names,names+["Outside option"]

def main(in_csv="input/data_yoghurt_clean.csv",
         out_elast="output/q21_nl_elasticities_city1p1.tex",
         out_div="output/q21_nl_diversion_city1p1.tex",
         zdist="distance",zdiesel="diesel",
         city=1,period=1):
    os.makedirs(os.path.dirname(out_elast),exist_ok=True)
    os.makedirs(os.path.dirname(out_div),exist_ok=True)
    df=pd.read_csv(in_csv); df=_prep(df,zdist,zdiesel)
    a,r=estimate_alpha_rho(df)
    E,rn,cn,D,rn2,cn2=elasticities_and_diversion(df,a,r,city,period)
    with open(out_elast,"w") as f:
        f.write(_booktabs_matrix(E,rn,cn,f"Nested-logit elasticities (City {city}, Period {period}).",f"tab:q21_elast"))
    with open(out_div,"w") as f:
        f.write(_booktabs_matrix(D,rn2,cn2,f"Nested-logit diversion ratios (City {city}, Period {period}).",f"tab:q21_div"))
    print("Saved:",out_elast,"and",out_div)

if __name__=="__main__":
    main()
