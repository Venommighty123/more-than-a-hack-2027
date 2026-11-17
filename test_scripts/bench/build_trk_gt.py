"""Tracking-based bench truth (like the organisers', who use corrected tracking).
Live frames: the team of the player the ball is on in Metrica's CLEAN tracking (ball within
TRK_R m of a player, the other team's nearest at least TRK_M m further, for at least TRK_RUN
frames in a row). Flights / loose / contested balls keep the last team (CHALLENGE.md).
After a stoppage the first team to control it owns the frames back to the restart.
D and X: copied from the event-based truth."""
import numpy as np, sys
TRK_R, TRK_M, TRK_RUN = 0.5, 0.5, 3
def build(d):
    b=d["ball"]; px,py=d["px"],d["py"]; team=np.asarray(d["team"]); gt=d["gt"]; n=len(b)
    ok=~np.isnan(b[:,0]); c=np.full(n,-1,np.int8)
    dd=np.hypot(px[ok]-b[ok,0:1],py[ok]-b[ok,1:2]); dd=np.where(np.isnan(dd),np.inf,dd)
    dh=dd[:,team==0].min(1); da=dd[:,team==1].min(1); cc=np.full(ok.sum(),-1,np.int8)
    cc[(dh<TRK_R)&(da>dh+TRK_M)]=0; cc[(da<TRK_R)&(dh>da+TRK_M)]=1; c[ok]=cc
    # keep only runs >= TRK_RUN
    out=gt.copy(); live=np.isin(gt,["H","A"])
    e=np.diff(np.r_[0,live.astype(int),0]); S=np.flatnonzero(e==1); E=np.flatnonzero(e==-1)
    for s,t in zip(S,E):
        cur=-1; run=0; prev=-1; pend=s; seg=np.full(t-s,-1,np.int8)
        for i in range(s,t):
            x=c[i]
            run = run+1 if (x>=0 and x==prev) else (1 if x>=0 else 0); prev=x
            if x>=0 and run>=TRK_RUN and x!=cur:
                start=i-TRK_RUN+1
                if cur==-1: seg[pend-s:start-s]=x
                cur=x
                seg[start-s:i-s+1]=x
                continue
            seg[i-s]=cur
        if cur==-1:
            seg[:]=np.where(gt[s:t]=="H",0,1)    # never controlled: keep event truth
        seg[seg==-1]=np.where(gt[s:t][seg==-1]=="H",0,1)
        out[s:t]=np.array(["H","A"])[seg]
    return out
if __name__=="__main__":
    from hutil import changes  # test_scripts/bench/hutil.py
    for g in ("g1","g2","g3"):
        d=np.load(f"bench/{g}_clean.npz"); gt=d["gt"]; tg=build(d); np.save(f"bench/{g}_gttrk.npy",tg)
        live=np.isin(gt,["H","A"])
        def nposs(x):
            n=0
            for _,lo,hi in d["blocks"]:
                seg=x[lo:hi+1]; ed=np.r_[0,np.flatnonzero(seg[1:]!=seg[:-1])+1]; n+=sum(seg[k] in "HA" for k in ed)
            return n
        print(f"{g}: live frames same team {np.mean(gt[live]==tg[live])*100:.1f}%; possessions event {nposs(gt)} vs tracking {nposs(tg)}; direct changes {sum(1 for c in changes(gt) if not c[2])} vs {sum(1 for c in changes(tg) if not c[2])}")
