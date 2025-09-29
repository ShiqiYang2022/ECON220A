# q20_nested_ivfe.py  (fixed: use N, not log N)
import os, numpy as np, pandas as pd, statsmodels.api as sm

NAMES={1:"Yoplait",2:"Chobani",3:"Dannon",4:"Stonyfield Farm",5:"Activia"}
NESTMAP={1:"sugary",2:"nosugar",3:"nosugar",4:"sugary",5:"sugary"}

def _booktabs(df,caption,label,colfmt=None):
    if colfmt is None: colfmt="l"+"c"*(df.shape[1]-1)
    body=df.to_latex(index=False,escape=False,column_format=colfmt,bold_rows=False)
    return "\\begin{table}[H]\n"+f"\\centering\n\\caption{{{caption}}}\n"+body+f"\\label{{{label}}}\n\\end{{table}}\n"

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
    t=(beta/se).flatten()
    return beta.flatten(), se.flatten(), t

def _prep(df, zdist, zdiesel):
    for c in ["price","share","weight","calories","sugar","protein",zdist,zdiesel]:
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

def _fes(df):
    d_prod=pd.get_dummies(df["product"],prefix="prod",drop_first=True)
    d_city=pd.get_dummies(df["city"],prefix="city",drop_first=True)
    d_per=pd.get_dummies(df["period"],prefix="per",drop_first=True)
    FE=pd.concat([d_prod,d_city,d_per],axis=1)
    FE=sm.add_constant(FE,has_constant="add")
    return FE

def estimate_alpha_rho(df):
    FE=_fes(df)
    X=pd.concat([df[["price","log_s_jg"]],FE],axis=1)
    Z=pd.concat([df[["iv_cost","iv_N"]],FE],axis=1)  
    X=X.replace([np.inf,-np.inf],np.nan).apply(pd.to_numeric,errors="coerce").astype(float)
    Z=Z.replace([np.inf,-np.inf],np.nan).apply(pd.to_numeric,errors="coerce").astype(float)
    y=pd.to_numeric(df["logit_dep_nl"].replace([np.inf,-np.inf],np.nan),errors="coerce").astype(float)
    ok=y.notna()&X.notna().all(axis=1)&Z.notna().all(axis=1)
    y,X,Z=y.loc[ok],X.loc[ok],Z.loc[ok]
    b,se,t=_ehw_2sls(y,X,Z)
    cols=X.columns.tolist()
    a=-float(b[cols.index("price")])
    r=float(b[cols.index("log_s_jg")])
    sa=float(se[cols.index("price")])
    sr=float(se[cols.index("log_s_jg")])
    return a,sa,r,sr

def main(in_csv="input/data_yoghurt_clean.csv",
         out_tex="output/q20_nested_ivfe.tex",
         zdist="distance", zdiesel="diesel"):
    os.makedirs(os.path.dirname(out_tex),exist_ok=True)
    df=pd.read_csv(in_csv)
    df=_prep(df,zdist,zdiesel)
    a,sa,r,sr=estimate_alpha_rho(df)
    tab=pd.DataFrame({
        "Variable":["Price ($\\alpha$)","$\\log s_{j|g}$ ($\\rho$)"],
        "Coef.":[-a, r],
        "Std. Err. (HC1)":[sa, sr]
    })
    for c in ["Coef.","Std. Err. (HC1)"]:
        tab[c]=tab[c].map(lambda x: f"{x:.4f}")
    with open(out_tex,"w") as f:
        f.write(_booktabs(tab,"Nested-logit IV with product/city/period FE (HC1).","tab:q20_nl_ivfe",colfmt="lcc"))
    print("alpha_hat =",a,"rho_hat =",r)
    print("Saved:",out_tex)

if __name__=="__main__":
    main()
