import os, numpy as np, pandas as pd, statsmodels.api as sm

NAMES={1:"Yoplait",2:"Chobani",3:"Dannon",4:"Stonyfield Farm",5:"Activia"}
NESTMAP={1:"sugary",2:"nosugar",3:"nosugar",4:"sugary",5:"sugary"}

def _booktabs(df,caption,label):
    body=df.to_latex(index=False,escape=False,column_format="lcccc",bold_rows=False)
    return "\\begin{table}[H]\n"+f"\\centering\n\\caption{{{caption}}}\n"+body+f"\\label{{{label}}}\n\\end{{table}}\n"

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

def _prep_nl(df,zdist,zdiesel):
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

def _prep_mnl(df,zdist,zdiesel):
    for c in ["price","share",zdist,zdiesel]:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    df["market"]=list(zip(df["city"],df["period"]))
    inside=df.groupby("market")["share"].sum().rename("inside_share_sum")
    df=df.merge(inside,on="market",how="left")
    eps=1e-12
    df["s0"]=1.0-df["inside_share_sum"]
    df["logit_dep"]=np.log(np.clip(df["share"],eps,None))-np.log(np.clip(df["s0"],eps,None))
    df["iv_cost"]=df[zdist]*df[zdiesel]
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
    b=_ehw_2sls(y,X,Z); cols=X.columns.tolist()
    return -float(b[cols.index("price")]), float(b[cols.index("log_s_jg")])

def estimate_alpha_mnl(df):
    FE=_fes(df)
    X=pd.concat([df[["price"]],FE],axis=1)
    Z=pd.concat([df[["iv_cost"]],FE],axis=1)
    X=X.replace([np.inf,-np.inf],np.nan).apply(pd.to_numeric,errors="coerce").astype(float)
    Z=Z.replace([np.inf,-np.inf],np.nan).apply(pd.to_numeric,errors="coerce").astype(float)
    y=pd.to_numeric(df["logit_dep"].replace([np.inf,-np.inf],np.nan),errors="coerce").astype(float)
    ok=y.notna()&X.notna().all(axis=1)&Z.notna().all(axis=1)
    y,X,Z=y.loc[ok],X.loc[ok],Z.loc[ok]
    b=_ehw_2sls(y,X,Z); cols=X.columns.tolist()
    return -float(b[cols.index("price")])

def owners_matrix(products,merged_groups=None):
    J=len(products); Om=np.eye(J)
    if merged_groups:
        idx={p:i for i,p in enumerate(products)}
        for grp in merged_groups:
            ids=[idx[p] for p in grp if p in idx]
            for i in ids:
                for j in ids:
                    Om[i,j]=1.0
    return Om

def nl_delta_tilde(m,alpha,r):
    s=m["share"].to_numpy(float); p=m["price"].to_numpy(float)
    s0=1.0-s.sum()
    gsum=m.groupby("nest")["share"].transform("sum").to_numpy(float)
    s_jg=np.clip(s/np.clip(gsum,1e-12,None),1e-12,1-1e-12)
    return np.log(s+1e-12)-np.log(max(1e-12,s0))+alpha*p-r*np.log(s_jg)

def nl_shares_from_delta(delta,r):
    mx=float(np.max(delta)); J=len(delta)
    nests=m_nest
    vals={}
    for g in np.unique(nests):
        idx=[i for i in range(J) if nests[i]==g]
        eg=np.exp((delta[idx]-mx)/(1.0-r))
        vals[g]=eg
    num=np.zeros(J)
    for g in np.unique(nests):
        idx=[i for i in range(J) if nests[i]==g]
        sg=vals[g].sum()
        num[idx]=vals[g]/sg
    den=1.0
    for g in np.unique(nests):
        den+=np.power(vals[g].sum(),1.0-r)
    s_g={}
    for g in np.unique(nests):
        s_g[g]=np.power(vals[g].sum(),1.0-r)/den
    s=np.zeros(J)
    for i in range(J):
        s[i]=num[i]*s_g[nests[i]]
    return s

def nl_J_matrix(s,s_jg,nests):
    J=len(s); M=np.zeros((J,J))
    for j in range(J):
        for k in range(J):
            if j==k:
                M[j,j]=s[j]*(1.0 - rho*(1.0 - s_jg[j]) - s[j])
            else:
                if nests[j]==nests[k]:
                    M[j,k]=s[j]*(s[k]-rho*s_jg[k])
                else:
                    M[j,k]=s[j]*s[k]
    return M

def nl_fixed_point(p0,c,alpha,r,delta_tilde,nests,owners,tol=1e-12,itmax=10000):
    p=p0.copy(); J=len(p)
    for _ in range(itmax):
        delta=delta_tilde-alpha*p
        egroups={}
        mx=float(np.max(delta))
        for g in np.unique(nests):
            idx=[i for i in range(J) if nests[i]==g]
            egroups[g]=np.exp((delta[idx]-mx)/(1.0-r))
        num=np.zeros(J); den=1.0
        for g in np.unique(nests):
            idx=[i for i in range(J) if nests[i]==g]
            Sg=egroups[g].sum()
            num[idx]=egroups[g]/Sg
            den+=np.power(Sg,1.0-r)
        s_g={g: np.power(egroups[g].sum(),1.0-r)/den for g in np.unique(nests)}
        s=np.zeros(J)
        for i in range(J): s[i]=num[i]*s_g[nests[i]]
        gsum=np.zeros(J)
        for g in np.unique(nests):
            idx=[i for i in range(J) if nests[i]==g]
            gsum[idx]=s[idx].sum()
        s_jg=np.clip(s/np.clip(gsum,1e-12,None),1e-12,1-1e-12)
        M=nl_J_matrix(s,s_jg,nests)
        Delta=alpha*(owners*M)
        mu=np.linalg.solve(Delta,s)
        p_new=c+mu
        if np.max(np.abs(p_new-p))<tol: return p_new
        p=p_new
    return p

def mnl_fixed_point(p0,c,alpha,delta_tilde,owners,tol=1e-12,itmax=10000):
    p=p0.copy(); J=len(p)
    for _ in range(itmax):
        delta=delta_tilde-alpha*p
        mx=float(np.max(delta))
        eg=np.exp(delta-mx)
        s=eg/(1.0+eg.sum())
        M=np.diag(s)-np.outer(s,s)
        Delta=alpha*(owners*(M))
        mu=np.linalg.solve(Delta,s)
        p_new=c+mu
        if np.max(np.abs(p_new-p))<tol: return p_new
        p=p_new
    return p

def main(in_csv="input/data_yoghurt_clean.csv",
         out_tex="output/q23_nl_merger_prices_city1p1.tex",
         zdist="distance",zdiesel="diesel",
         city=1,period=1,merge_group=(2,3)):
    os.makedirs(os.path.dirname(out_tex),exist_ok=True)
    df=pd.read_csv(in_csv)
    dfnl=_prep_nl(df.copy(),zdist,zdiesel); a,r=estimate_alpha_rho(dfnl)
    dfm=_prep_mnl(df.copy(),zdist,zdiesel); a_std=estimate_alpha_mnl(dfm)
    m=dfnl[(dfnl["city"]==city)&(dfnl["period"]==period)].copy().sort_values("product")
    global m_nest, rho
    m_nest=m["nest"].tolist(); rho=r
    p_obs=m["price"].to_numpy(float); s=m["share"].to_numpy(float)
    gsum=m.groupby("nest")["share"].transform("sum").to_numpy(float)
    s_jg=np.clip(s/np.clip(gsum,1e-12,None),1e-12,1-1e-12)
    delta_tilde_nl=np.log(s+1e-12)-np.log(max(1e-12,1.0-s.sum()))+a*p_obs-r*np.log(s_jg)
    c_nl=p_obs-1.0/(a*(1.0 - r*(1.0 - s_jg) - s))
    prods=m["product"].astype(int).tolist()
    names=[NAMES.get(p,str(p)) for p in prods]
    Om_pre=np.eye(len(prods))
    Om_post=owners_matrix(prods,merged_groups=[merge_group])
    p_pre=nl_fixed_point(p_obs.copy(),c_nl,a,r,delta_tilde_nl,m_nest,Om_pre)
    p_post=nl_fixed_point(p_obs.copy(),c_nl,a,r,delta_tilde_nl,m_nest,Om_post)
    mm=dfm[(dfm["city"]==city)&(dfm["period"]==period)].copy().sort_values("product")
    p_obs_std=mm["price"].to_numpy(float); s_std=mm["share"].to_numpy(float)
    delta_tilde_std=np.log(s_std+1e-12)-np.log(max(1e-12,1.0-s_std.sum()))+a_std*p_obs_std
    c_std=p_obs_std-1.0/(a_std*(1.0 - s_std))
    Om_post_std=owners_matrix(mm["product"].astype(int).tolist(),merged_groups=[merge_group])
    p_post_std=mnl_fixed_point(p_obs_std.copy(),c_std,a_std,delta_tilde_std,Om_post_std)
    out=pd.DataFrame({"Product":names,"Old price":p_pre,"New price (NL merge)":p_post,"New price (MNL merge)":p_post_std})
    with open(out_tex,"w") as f:
        f.write(_booktabs(out,f"Merger counterfactual (Chobani+Dannon), City {city}, Period {period}.","tab:q23_nl_merge"))
    print("alpha_nl =",a,"rho =",r,"alpha_mnl =",a_std)
    print("Saved:",out_tex)

if __name__=="__main__":
    main()
