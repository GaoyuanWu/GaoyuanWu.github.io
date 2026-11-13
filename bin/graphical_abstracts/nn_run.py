import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import JaxSSO.model as Model
from JaxSSO.SSO_model import NodeParameter, SSO_model
n=16; L=6.0
g=np.linspace(0,L,n+1); X,Y=np.meshgrid(g,g); xs=X.ravel(); ys=Y.ravel(); N=xs.size
idx=lambda i,j:i*(n+1)+j
cnct=np.array([[idx(i,j),idx(i,j+1),idx(i+1,j+1),idx(i+1,j)] for i in range(n) for j in range(n)])
# supports: 2x2 node patch at each corner
sup=[idx(i,j) for i in (0,1,n-1,n) for j in (0,1,n-1,n)]
sup=np.array(sorted(set(sup))); design=np.setdiff1d(np.arange(N),sup)
z0=0.1*(-(xs-L/2)**2+(L/2)**2-(ys-L/2)**2+(L/2)**2)  # shallow dome like the notebook
z0[sup]=0.0
t,E,nu,P=0.1,3e10,0.2,2e3
m=Model.Model()
for k in range(N):
    m.add_node(k,xs[k],ys[k],z0[k])
    if k in sup: m.add_support(k,[1,1,1,1,1,1])
    else: m.add_nodal_load(k,[0,0,-P,0,0,0])
for e,c in enumerate(cnct): m.add_quad(e,*map(int,c),t,E,nu)
m.model_ready()
sso=SSO_model(m)
for k in design: sso.add_nodeparameter(NodeParameter(int(k),2))
sso.initialize_parameters_values(); sso.set_objective('strain energy')
# NN features: normalized x, y, distance-to-supports centrality
bd=np.array([np.min(np.hypot(xs[k]-xs[sup],ys[k]-ys[sup])) for k in range(N)]); bd/=bd.max()
feat=jnp.array(np.c_[xs/L*2-1, ys/L*2-1, bd][design])
key=jax.random.PRNGKey(0); sizes=[3,32,32,32,1]; params=[]
for a,b in zip(sizes[:-1],sizes[1:]):
    key,k1=jax.random.split(key); params.append((jax.random.normal(k1,(a,b))*np.sqrt(1/a), jnp.zeros(b)))
def net(p):
    h=feat
    for W,b in p[:-1]: h=jnp.tanh(h@W+b)
    W,b=p[-1]; return 3.0*jax.nn.sigmoid((h@W+b)[:,0])  # heights in (0,3) m
# pretrain to the initial dome so training starts from the "initial structure"
zt=jnp.array(z0[design])
lossfit=jax.jit(jax.value_and_grad(lambda p: jnp.mean((net(p)-zt)**2)))
def adam(params,grads,st,lr,b1=0.9,b2=0.999):
    mm,vv,t=st; t+=1
    mm=jax.tree_util.tree_map(lambda m_,g_:b1*m_+(1-b1)*g_,mm,grads)
    vv=jax.tree_util.tree_map(lambda v_,g_:b2*v_+(1-b2)*g_**2,vv,grads)
    params=jax.tree_util.tree_map(lambda p_,m_,v_:p_-lr*(m_/(1-b1**t))/(jnp.sqrt(v_/(1-b2**t))+1e-8),params,mm,vv)
    return params,(mm,vv,t)
zeros=lambda p: jax.tree_util.tree_map(jnp.zeros_like,p)
st=(zeros(params),zeros(params),0)
for i in range(1500):
    l,gr=lossfit(params); params,st=adam(params,gr,st,3e-3)
print('pretrain mse',float(l))
st=(zeros(params),zeros(params),0); Ch=[];Zh=[]
for it in range(150):
    zd,vjp=jax.vjp(net,params)
    sso.update_nodeparameter(np.asarray(zd))
    C,dC=sso.value_grad_params(which_solver='sparse',enforce_scipy_sparse=True)
    Ch.append(float(C)); Zh.append(np.asarray(zd))
    (gp,)=vjp(jnp.array(np.asarray(dC,dtype=float)))
    params,st=adam(params,gp,st,4e-3)
    if it%25==0: print(it,float(C),float(zd.max()))
np.savez('ga/nn_hist.npz',xs=xs,ys=ys,cnct=cnct,sup=sup,design=design,C=np.array(Ch),Z=np.array(Zh),feat=np.array(feat))
print('final',Ch[0],Ch[-1])
