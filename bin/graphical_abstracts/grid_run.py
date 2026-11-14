import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import JaxSSO.model as Model
from JaxSSO.SSO_model import NodeParameter, SSO_model

n=12; L=20.0
xs1=np.linspace(0,L,n+1)
X,Y=np.meshgrid(xs1,xs1); xs=X.ravel(); ys=Y.ravel(); N=xs.size
idx=lambda i,j:i*(n+1)+j
edges=[]
for i in range(n+1):
    for j in range(n+1):
        if j<n: edges.append((idx(i,j),idx(i,j+1)))
        if i<n: edges.append((idx(i,j),idx(i+1,j)))
edges=np.array(edges)
bnd=np.array([k for k in range(N) if xs[k] in (0,L) or ys[k] in (0,L)])
free=np.setdiff1d(np.arange(N),bnd)
P=10e3; H=5.0
E,G,A,I,J=210e9,81e9,0.006,3e-5,6e-5

def build(z):
    m=Model.Model()
    for k in range(N):
        m.add_node(k,xs[k],ys[k],z[k])
        if k in bnd: m.add_support(k,[1,1,1,0,0,0])
        else: m.add_nodal_load(k,[0,0,-P,0,0,0])
    for e,(a,b) in enumerate(edges): m.add_beamcol(e,int(a),int(b),E,G,I,I,J,A)
    m.model_ready()
    s=SSO_model(m)
    for k in free: s.add_nodeparameter(NodeParameter(int(k),2))
    s.initialize_parameters_values(); s.set_objective('strain energy')
    return s
sso=build(np.zeros(N)+0.0)
def C_and_grad(zfree):
    sso.update_nodeparameter(zfree)
    C,g=sso.value_grad_params(which_solver='sparse',enforce_scipy_sparse=True)
    return float(C),np.asarray(g,dtype=float)

# ---- FSO: optimize z directly (projected normalized GD) ----
rng=np.random.default_rng(0)
z=rng.uniform(0.0,0.05,free.size)
fso_C=[];fso_Z=[]
for it in range(80):
    C,g=C_and_grad(z); fso_C.append(C); fso_Z.append(z.copy())
    z=np.clip(z-0.25*g/np.linalg.norm(g)*np.sqrt(free.size)*0.12,0,H)
print('FSO',fso_C[0],fso_C[-1],z.max())

# ---- constrained FDM: optimize force densities ----
Cmat=np.zeros((len(edges),N)); Cmat[np.arange(len(edges)),edges[:,0]]=1; Cmat[np.arange(len(edges)),edges[:,1]]=-1
Cf=jnp.array(Cmat[:,free])
def fdm(logq):
    q=jnp.exp(logq)
    D=Cf.T@(q[:,None]*Cf)
    zf=jnp.linalg.solve(D,-(-P)*jnp.ones(free.size))  # z up for downward load
    return jnp.minimum(zf, H)  # height cap
q0=np.log(np.ones(len(edges))*P*1.0)  # gives modest initial rise
zf0=np.array(fdm(q0)); scale=np.log(zf0.max()/1.0); q0=q0+scale
lq=q0.copy(); fdm_C=[];fdm_Z=[]
for it in range(80):
    zf,vjp=jax.vjp(fdm,jnp.array(lq))
    C,g=C_and_grad(np.array(zf)); fdm_C.append(C); fdm_Z.append(np.array(zf))
    (gq,)=vjp(jnp.array(g)); gq=np.array(gq)
    lq=lq-0.03*gq/np.linalg.norm(gq)*np.sqrt(len(edges))*0.1
    if it%10==0: print(it,C,np.array(zf).max())
print('FDM',fdm_C[0],fdm_C[-1],np.array(fdm(jnp.array(lq))).max())
np.savez('ga/grid_hist.npz',xs=xs,ys=ys,edges=edges,free=free,fso_C=fso_C,fso_Z=fso_Z,fdm_C=fdm_C,fdm_Z=fdm_Z,H=H)
