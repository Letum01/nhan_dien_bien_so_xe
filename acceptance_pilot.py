"""Deterministic edit breakdown for recognition evaluation."""

def edit_counts(ref,pred):
    """Minimum edit alignment; deterministic tie order substitution/deletion/insertion."""
    dp=[[(0,0,j) for j in range(len(pred)+1)]]
    for i,a in enumerate(ref,1):
        current=[(0,i,0)]
        for j,b in enumerate(pred,1):
            s,d,n=dp[-1][j-1]; x,y,z=dp[-1][j];u,v,w=current[-1]
            current.append(min([(s+int(a!=b),d,n),(x,y+1,z),(u,v,w+1)],key=sum))
        dp.append(current)
    return dict(zip(('substitutions','deletions','insertions'),dp[-1][-1]))
