import os, numpy as np, pandas as pd, statsmodels.api as sm

NAMES={1:"Yoplait",2:"Chobani",3:"Dannon",4:"Stonyfield Farm",5:"Activia"}
NESTMAP={1:"sugary",2:"nosugar",3:"nosugar",4:"sugary",5:"sugary"}

def _booktabs(df,caption,label):
    head="\\begin{table}[H]\n\\centering\n"
    head+=f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
    head+="\\begin{tabular}{lcccc}\n\\toprule\n"
    head+="Product & Price & Share & $1/\\{\\hat\\alpha[1-\\hat\\rho(1-s_{j|g})-s_j]\\}$ & Marginal cost $c_j$ \\\\\n\\midrule\n"
    body="\n".join(f"{r.Product} & {r.Price:.4f} & {r.Share:.4f} & {r.Markup:.4f} & {r.MC:.4f} \\\\" for _,r in df.iterrows())
    tail="\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return head+body+tail

def _fes(df):
    d_prod=pd.get_dummies(df["product"],prefix="prod",drop_first=True)
    d_city=pd.get_dummies(df["city"],prefix="city",drop_first=True)
    d_per=pd.get_dummies(df["period"],prefix="per",drop_first=True)
    FE=pd.concat([d_prod,d_city,d_per],axis=1)
    FE=sm.add_constant(FE,has_constant="add")
    return FE

def _ehw_2sls(y,X,Z):
    if hasattr(y,"values"): y=y.values
    if hasattr(X,"values"): X=X.values
    if hasattr(Z,"values"): Z=Z.values
    y=y.reshape(-1,1).astype(float); X=X.astype(float); Z=Z.astype(float)
    ZZ=Z.T@Z; ZZinv=np.linalg.inv(ZZ); XZ=X.T@Z; ZY=Z.T@y
    beta=np.linalg.inv(XZ@ZZinv@XZ.T)@(XZ@ZZinv@ZY)
    return beta.flatten()

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
    df["iv_lN"]=np.log(ncount.astype(float))
    return df

def estimate_alpha_rho(df):
    FE=_fes(df)
    X=pd.concat([df[["price","log_s_jg"]],FE],axis=1)
    Z=pd.concat([df[["iv_cost","iv_lN"]],FE],axis=1)
    X=X.replace([np.inf,-np.inf],np.nan).apply(pd.to_numeric,errors="coerce").astype(float)
    Z=Z.replace([np.inf,-np.inf],np.nan).apply(pd.to_numeric,errors="coerce").astype(float)
    y=pd.to_numeric(df["logit_dep_nl"].replace([np.inf,-np.inf],np.nan),errors="coerce").astype(float)
    ok=y.notna()&X.notna().all(axis=1)&Z.notna().all(axis=1)
    y,X,Z=y.loc[ok],X.loc[ok],Z.loc[ok]
    b=_ehw_2sls(y,X,Z)
    cols=X.columns.tolist()
    a=-float(b[cols.index("price")]); r=float(b[cols.index("log_s_jg")])
    return a,r

def main(in_csv="input/data_yoghurt_clean.csv",
         out_tex="output/q22_nl_costs_city1p1.tex",
         zdist="distance",zdiesel="diesel",
         city=1,period=1):
    os.makedirs(os.path.dirname(out_tex),exist_ok=True)
    df=pd.read_csv(in_csv); df=_prep(df,zdist,zdiesel)
    a,r=estimate_alpha_rho(df)
    m=df[(df["city"]==city)&(df["period"]==period)].copy().sort_values("product")
    m["Product"]=m["product"].map(NAMES).fillna(m["product"].astype(str))
    s=m["share"].to_numpy(float); p=m["price"].to_numpy(float)
    gsum=m.groupby("nest")["share"].transform("sum").to_numpy(float)
    s_jg=np.clip(s/np.clip(gsum,1e-12,None),1e-12,1-1e-12)
    markup=1.0/(a*(1.0 - r*(1.0 - s_jg) - s))
    mc=p-markup
    tbl=pd.DataFrame({"Product":m["Product"],"Price":p,"Share":s,"Markup":markup,"MC":mc})
    with open(out_tex,"w") as f:
        f.write(_booktabs(tbl,f"Nested-logit recovered marginal costs (City {city}, Period {period}).","tab:q22_nl_costs"))
    print("alpha =",a,"rho =",r)
    print("Saved:",out_tex)

if __name__=="__main__":
    main()
