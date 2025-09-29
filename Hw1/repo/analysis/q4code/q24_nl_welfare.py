import os, numpy as np, pandas as pd, statsmodels.api as sm

NESTMAP={1:"sugary",2:"nosugar",3:"nosugar",4:"sugary",5:"sugary"}

def _booktabs(df,caption,label,colfmt=None):
    if colfmt is None: colfmt="lcc"
    body=df.to_latex(index=False,escape=False,column_format=colfmt,bold_rows=False)
    return "\\begin{table}[H]\n\\centering\n"+f"\\caption{{{caption}}}\n"+body+f"\\label{{{label}}}\n\\end{{table}}\n"

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
    df["s0"]=1.0-df["inside_share_sum"]
    eps=1e-12
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

def nl_shares_from_delta(delta,nests,r):
    mx=float(np.max(delta)); J=len(delta)
    egroups={}
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
    return s

def nl_J_matrix(s,s_jg,nests,r):
    J=len(s); M=np.zeros((J,J))
    for j in range(J):
        for k in range(J):
            if j==k:
                M[j,j]=s[j]*(1.0 - r*(1.0 - s_jg[j]) - s[j])
            else:
                if nests[j]==nests[k]:
                    M[j,k]=s[j]*(s[k]-r*s_jg[k])
                else:
                    M[j,k]=s[j]*s[k]
    return M

def nl_fixed_point(p0,c,alpha,r,delta_tilde,nests,owners,tol=1e-12,itmax=10000):
    p=p0.copy(); J=len(p)
    for _ in range(itmax):
        delta=delta_tilde-alpha*p
        s=nl_shares_from_delta(delta,nests,r)
        gsum=np.zeros(J)
        for g in np.unique(nests):
            idx=[i for i in range(J) if nests[i]==g]
            gsum[idx]=s[idx].sum()
        s_jg=np.clip(s/np.clip(gsum,1e-12,None),1e-12,1-1e-12)
        M=nl_J_matrix(s,s_jg,nests,r)
        Delta=alpha*(owners*M)
        mu=np.linalg.solve(Delta,s)
        p_new=c+mu
        if np.max(np.abs(p_new-p))<tol: return p_new
        p=p_new
    return p

def mnl_fixed_point(p0,c,alpha,delta_tilde,owners,tol=1e-12,itmax=10000):
    p=p0.copy()
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

def cs_nested(delta_tilde,p,alpha,r,nests):
    delta=delta_tilde-alpha*p
    mx=float(np.max(delta)); J=len(delta)
    tot=1.0
    for g in np.unique(nests):
        idx=[i for i in range(J) if nests[i]==g]
        Sg=np.sum(np.exp((delta[idx]-mx)/(1.0-r)))
        tot+=np.power(Sg,1.0-r)
    return np.log(tot)+mx/alpha

def cs_mnl(delta_tilde,p,alpha):
    delta=delta_tilde-alpha*p
    mx=float(np.max(delta))
    iv=np.log(1.0+np.sum(np.exp(delta-mx)))+mx
    return iv/alpha

def main(in_csv="input/data_yoghurt_clean.csv",
         out_tex="output/q24_nl_welfare_city1p1.tex",
         zdist="distance",zdiesel="diesel",
         city=1,period=1,merge_group=(2,3)):
    os.makedirs(os.path.dirname(out_tex),exist_ok=True)
    df=pd.read_csv(in_csv)
    dfnl=_prep_nl(df.copy(),zdist,zdiesel); a,r=estimate_alpha_rho(dfnl)
    dfm=_prep_mnl(df.copy(),zdist,zdiesel); a_std=estimate_alpha_mnl(dfm)
    m=dfnl[(dfnl["city"]==city)&(dfnl["period"]==period)].copy().sort_values("product")
    nests=m["nest"].tolist()
    p_obs=m["price"].to_numpy(float); s=m["share"].to_numpy(float)
    S0=max(1e-12,1.0-s.sum())
    gsum=m.groupby("nest")["share"].transform("sum").to_numpy(float)
    s_jg=np.clip(s/np.clip(gsum,1e-12,None),1e-12,1-1e-12)
    delta_tilde_nl=np.log(s+1e-12)-np.log(S0)+a*p_obs-r*np.log(s_jg)
    c_nl=p_obs-1.0/(a*(1.0 - r*(1.0 - s_jg) - s))
    Om_pre=np.eye(len(p_obs)); Om_post=Om_pre.copy()
    idx={int(p):i for i,p in enumerate(m["product"].tolist())}
    ids=[idx[p] for p in merge_group if p in idx]
    for i in ids:
        for j in ids: Om_post[i,j]=1.0
    p_pre=nl_fixed_point(p_obs.copy(),c_nl,a,r,delta_tilde_nl,nests,Om_pre)
    p_post=nl_fixed_point(p_obs.copy(),c_nl,a,r,delta_tilde_nl,nests,Om_post)
    s_pre=nl_shares_from_delta(delta_tilde_nl-a*p_pre,nests,r)
    s_post=nl_shares_from_delta(delta_tilde_nl-a*p_post,nests,r)
    cs_pre=cs_nested(delta_tilde_nl,p_pre,a,r,nests)
    cs_post=cs_nested(delta_tilde_nl,p_post,a,r,nests)
    spend_pre=np.sum(p_pre*s_pre)
    spend_post=np.sum(p_post*s_post)
    mm=dfm[(dfm["city"]==city)&(dfm["period"]==period)].copy().sort_values("product")
    p_obs_std=mm["price"].to_numpy(float); s_std=mm["share"].to_numpy(float)
    delta_tilde_std=np.log(s_std+1e-12)-np.log(max(1e-12,1.0-s_std.sum()))+a_std*p_obs_std
    c_std=p_obs_std-1.0/(a_std*(1.0 - s_std))
    Om_post_std=np.eye(len(p_obs_std))
    idx2={int(p):i for i,p in enumerate(mm["product"].tolist())}
    ids2=[idx2[p] for p in merge_group if p in idx2]
    for i in ids2:
        for j in ids2: Om_post_std[i,j]=1.0
    p_post_std=mnl_fixed_point(p_obs_std.copy(),c_std,a_std,delta_tilde_std,Om_post_std)
    cs_pre_std=cs_mnl(delta_tilde_std,p_obs_std,a_std)
    cs_post_std=cs_mnl(delta_tilde_std,p_post_std,a_std)
    out=pd.DataFrame({
        "Metric":["CS per capita","Total expend. per capita","Difference (post-pre)","CS change as % of pre expend."],
        "Nested Logit":[cs_pre,spend_pre,cs_post-cs_pre,100.0*(cs_post-cs_pre)/spend_pre],
        "Standard Logit":[cs_pre_std,np.sum(p_obs_std*(np.exp(delta_tilde_std-a_std*p_obs_std)/(1.0+np.sum(np.exp(delta_tilde_std-a_std*p_obs_std))))),cs_post_std-cs_pre_std,100.0*(cs_post_std-cs_pre_std)/np.sum(p_obs_std*(np.exp(delta_tilde_std-a_std*p_obs_std)/(1.0+np.sum(np.exp(delta_tilde_std-a_std*p_obs_std))))) ]
    }).round(6)
    with open(out_tex,"w") as f:
        f.write(_booktabs(out,f"Welfare before vs. after merger (City {city}, Period {period}).","tab:q24_nl_welfare"))
    print("Saved:",out_tex)

if __name__=="__main__":
    main()
