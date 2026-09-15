import json, pathlib, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=pathlib.Path('/Users/xangma/repos/torchwave')
source=pathlib.Path('/tmp/torchwave-qnm-independent-full')
out=root/'docs/reviews/2026-09-12-clean-room/verification/pass-7'
report=json.loads((source/'result.json').read_text())
fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
for ax,(ell,m) in zip(axes.flat,[(2,2),(2,1),(3,3),(3,2),(4,4),(2,-2)]):
    for n in (range(3) if m>0 else [0]):
        key=f'{ell},{m},{n}'
        count=report['modes'][key]['training_count']
        rows=json.loads((source/f'{key}-training-{count}.json').read_text())
        x=np.array([-np.log10(1-r['spin']) for r in rows])
        y=np.array([-r['omega'][1] for r in rows])
        ax.semilogy(x,y,label=f'n={n}')
    ax.set(title=f'(ell, m) = ({ell}, {m})',xlabel='-log10(1-a)',ylabel='-Im(M omega)')
    ax.grid(alpha=.2); ax.legend(frameon=False)
fig.suptitle('Kerr damping branches from our independent equation solves')
fig.savefig(out/'qnm-branches.png',dpi=180)
plt.close(fig)
labels=list(report['modes']); x=np.arange(len(labels))
a=np.array([report['modes'][m]['holdout']['max_abs'] for m in labels])
r=np.array([report['modes'][m]['holdout']['max_relative'] for m in labels])
fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,constrained_layout=True)
for ax,values,gate,ylabel in zip(axes,[a,r],[1e-12,1e-10],['Absolute error','Relative error']):
    ax.semilogy(x,values[:,0],'o',label='Re(M omega)')
    ax.semilogy(x,values[:,1],'s',label='-Im(M omega)')
    ax.axhline(gate,color='black',ls='--',label='Required bound')
    ax.set(ylabel=ylabel); ax.grid(alpha=.2); ax.legend(frameon=False,ncol=3)
axes[-1].set_xticks(x,labels,rotation=45,ha='right')
axes[-1].set_xlabel('Mode ell,m,n')
fig.suptitle('Interpolation checked against stricter equation solves')
fig.savefig(out/'qnm-validation.png',dpi=180)
plt.close(fig)
print('Saved branch and validation figures.')
