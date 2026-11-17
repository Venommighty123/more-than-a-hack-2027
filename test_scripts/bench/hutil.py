import numpy as np
def changes(lab):
    """(frame, new team, via_D) for every H<->A change; vectorised."""
    lab=np.asarray(lab); ha=np.isin(lab,["H","A"])
    idx=np.flatnonzero(ha); l=lab[idx]
    sw=np.flatnonzero(l[1:]!=l[:-1])+1
    return [(int(idx[i]),str(l[i]),bool(idx[i]-idx[i-1]>1)) for i in sw]
