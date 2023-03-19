import math
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from numpy import random
from scipy import stats
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform

is_gpu = torch.cuda.is_available()

device = torch.device('cuda' if is_gpu else 'cpu')
print(device)
print(torch.__version__)

def setup_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

#key parameters
test = 'maze2'
r = 1024
R = 4096
epoch = 1000
lr = 10**(-1)
d,L,M = 2,2,5 # d is the dimension of space, L is the width of network V.
T0, T = 0., 1.
N = 2*5  # N must be an even number because we use Simpson quadrature formula
h = (T-T0)/N
sigma0, sigma1 = 0.3,0.3 ###1.0,1.0
var0 = sigma0*torch.eye(d,device=device)
var1 = sigma1*torch.eye(d,device=device)
pi = torch.Tensor([math.pi]).to(device)
hidden = 5
setup_seed(0)

def mean(firstwo=[0., 0.]):
    E = torch.zeros(d,device=device)
    E[0:2] = torch.tensor(firstwo)
    return E

if test == 'test4':
    lam_FP = 500
    lam_KL = 10
    ###test4
    mean0 = mean(firstwo=[-3., 0.])
    mean1 = mean(firstwo=[3., 0.])
    mean_Q = torch.tensor([0.,0.], device=device)
    var_Q = torch.Tensor([[0.5,0],[0,1]]).to(device)
    def Q(x):
        return MultiNormal(x[:,0:2], mean_Q, var_Q)
    
elif test == 'test5':
    lam_FP = 500
    lam_KL = 10
    ###test5
    mean0 = mean(firstwo=[-4., 0.])
    mean1 = mean(firstwo=[4., 0.])
    mean_Q1 = torch.tensor([0.,-2.], device=device)
    var_Q1 = torch.Tensor([[0.1,0],[0,1]]).to(device)
    mean_Q2 = torch.tensor([0.,2.], device=device)
    var_Q2 = torch.Tensor([[0.1,0],[0,1]]).to(device)

    def Q(x):
        return (1/2)*MultiNormal(x[:,0:2], mean_Q1, var_Q1) + (1/2)*MultiNormal(x[:,0:2], mean_Q2, var_Q2)
    
elif test == 'maze1':
    lam_FP = 50000
    lam_KL = 100
   ###maze1
    mean0 = mean(firstwo=[-4.,-4.])
    mean1 = mean(firstwo=[4.,4.])
    
    def pp(x):
        return 0.1*x[:,0]+0.4
    
    def Q(x):
        left, right = -3.0, 10.0
        up, low = 1.0,-1.0
        m_x = Uniform(torch.tensor([left]), torch.tensor([right]))
        m_y = Uniform(torch.tensor([low]), torch.tensor([up]))
        p = m_x.log_prob(x[:,0]).exp()*m_y.log_prob(x[:,1]).exp()
        S = (right-left)*(up-low)
        return p*S*pp(x)
    
elif test == 'maze2':
    lam_FP = 5000
    lam_KL = 100
   ##maze2
    mean0 = mean(firstwo=[0.,-4.])
    mean1 = mean(firstwo=[0.,4.])
    
    def pp(x):
        return -0.1*x[:,0]+0.3
    
    def Q(x):
        left, right = -10.0, 2.0
        up, low = 1.0, -1.0
        m_x = Uniform(torch.tensor([left]), torch.tensor([right]))
        m_y = Uniform(torch.tensor([low]), torch.tensor([up]))
        p = m_x.log_prob(x[:,0]).exp()*m_y.log_prob(x[:,1]).exp()
        S = (right-left)*(up-low)
        return p*S*pp(x)
    
elif test == 'maze3':
    lam_FP = 5000
    lam_KL = 100
   ###maze3
    sigma0, sigma1 = 1.0,1.0
    var0 = sigma0*torch.eye(d,device=device)
    var1 = sigma1*torch.eye(d,device=device)
    
    mean0 = mean(firstwo=[0.,-4.])
    mean1 = mean(firstwo=[0.,4.])
    
    def pp(x):
        return 0.1*x[:,0].abs()
    
    def Q(x):
        left1, right1 = -10.0, -0.5
        up1, low1 = 1.0, -1.0
        left2, right2 = 0.5, 10.0
        up2, low2 = 1.0, -1.0
        m_x1 = Uniform(torch.tensor([left1]), torch.tensor([right1]))
        m_y1 = Uniform(torch.tensor([low1]), torch.tensor([up1]))
        p1 = m_x1.log_prob(x[:,0]).exp()*m_y1.log_prob(x[:,1]).exp()
        m_x2 = Uniform(torch.tensor([left2]), torch.tensor([right2]))
        m_y2 = Uniform(torch.tensor([low2]), torch.tensor([up2]))
        p2 = m_x2.log_prob(x[:,0]).exp()*m_y2.log_prob(x[:,1]).exp()
        p = p1+p2
        S1 = (right1-left1)*(up1-low1)
        return p*S1*pp(x)    
    
elif test == 'maze4':
    lam_FP = 5
    lam_KL = 100
    ###maze4
    mean0 = mean(firstwo=[-3.,-3.])
    mean1 = mean(firstwo=[3.,3.])
    
    def pp(x):
        
        m = int(ite/100)   
        if m > 6:
            m = 6
            #print('m:',m)
        
        return 0.1*(10**m)*x[:,0].abs()
    
    def Q(x):
        left1, right1 = -10.0, -1.0
        up1, low1 = 1.5, -0.5
        left2, right2 = 1.0, 10.0
        up2, low2 = 0.5, -1.5
        m_x1 = Uniform(torch.tensor([left1]), torch.tensor([right1]))
        m_y1 = Uniform(torch.tensor([low1]), torch.tensor([up1]))
        p1 = m_x1.log_prob(x[:,0]).exp()*m_y1.log_prob(x[:,1]).exp()
        m_x2 = Uniform(torch.tensor([left2]), torch.tensor([right2]))
        m_y2 = Uniform(torch.tensor([low2]), torch.tensor([up2]))
        p2 = m_x2.log_prob(x[:,0]).exp()*m_y2.log_prob(x[:,1]).exp()
        p = p1+p2
        S1 = (right1-left1)*(up1-low1)
        return p*S1*pp(x)
    
print('test:', test,'d:',d, 'lam_KL:',lam_KL,'lam_FP:',lam_FP,'r:',r,'L:',L,'M:',M,'hidden:',hidden)

def MultiNormal(x, mu, var):  #x (r,d)
    return MultivariateNormal(mu, var).log_prob(x).exp()

def lnrho0(x): #x(r,d) out(r)
    return MultivariateNormal(mean0, var0).log_prob(x)

def lnrho1(x): #x(r,d) out(r)
    return MultivariateNormal(mean1, var1).log_prob(x)


def phi(s,h):
    output = torch.relu(s+h)-2*torch.relu(s)+torch.relu(s-h)
    output = M/(T-T0)*output
    return output

def Phi(t):
    output = t*torch.ones(M+1,device=device)-torch.linspace(T0,T,M+1,device=device)
    return phi(output,(T-T0)/M)

class Block_v(nn.Module):
    def __init__(self):
        super(Block_v, self).__init__()
        self.linear1 = nn.Linear(d,hidden)
        self.linear2 = nn.Linear(hidden,d)

    def forward(self, x): # x(r,d)
        out1 = self.linear1(x)
        out2 = torch.tanh(out1)
        out = self.linear2(out2)
        return out #(r,d)

class Blocks_v(nn.Module):
    def __init__(self):
        super(Blocks_v, self).__init__()
        self.linears_v = nn.ModuleList([Block_v() for i in range(L)])

    def forward(self, x):
        y = torch.zeros_like(x)
        for i, l in enumerate(self.linears_v):
            y = y + l(x)
        return y

class Module_v(nn.Module):
    def __init__(self):
        super(Module_v, self).__init__()
        self.lists_v = nn.ModuleList([Blocks_v() for i in range(M+1)])

    def forward(self, x,t):
        y = torch.zeros_like(x)
        for i, l in enumerate(self.lists_v):
            y = y+l(x)*Phi(t)[i]
        return y

def diff1_tanh(x):
    output = 1-torch.tanh(x)**2
    return output

def Get_divV(v, x ,t):#x(r,d)
    r = x.size()[0]
    W1 = list(v.parameters())[0::4]
    b1 = list(v.parameters())[1::4]
    W2 = list(v.parameters())[2::4]
    WW1 = nn.utils.parameters_to_vector(W1).view(L*(M+1),d*hidden).view(M+1,L,hidden,d)
    WW2 = nn.utils.parameters_to_vector(W2).view(L*(M+1),d*hidden).view(M+1,L,d,hidden)
    bb1 = nn.utils.parameters_to_vector(b1).view(L*(M+1),hidden).view(M+1,L,hidden)
    C = diff1_tanh(torch.tensordot(WW1,x,dims=([-1],[-1]))\
                  +torch.unsqueeze(bb1,-1).expand(M+1,L,hidden,r))#(M+1,L,hidden,r)
    output = torch.einsum('ilkj,iljs,iljk,i->s', [WW2,C,WW1,Phi(t)]) #(r)
    return output


###  The RK-Block for solving X(t_{n})
class RK4_block_X(nn.Module):
    def __init__(self, module_v):
        super(RK4_block_X, self).__init__()
        self.v = module_v

    def forward(self, x, t, step=h/4):
        K1 = self.v(x,t)
        K2 = self.v(x+step/2*K1,t+step/2)
        K3 = self.v(x+step/2*K2,t+step/2)
        K4 = self.v(x+step*K3,t+step)
        y = x+step/6*(K1+2*K2+2*K3+K4)
        return y

class RK4_block_lnRo(nn.Module):
    def __init__(self, module_v):
        super(RK4_block_lnRo, self).__init__()
        self.v = module_v

    def forward(self, lnRo, x0,x1,x2, t, step = h/2):  #x (3,r,d)
        divV1 = Get_divV(self.v,x0,t)
        divV2 = Get_divV(self.v,x1,t+step/2)
        divV3 = Get_divV(self.v,x2,t+step)
        K1 = -divV1
        K2 = -divV2
        K3 = -divV2
        K4 = -divV3
        y = lnRo+step/6*(K1+2*K2+2*K3+K4)

        return y


def Get_gradV(v, x ,t): #x(r,d)
    r = x.size()[0]
    W1 = list(v.parameters())[0::4]
    b1 = list(v.parameters())[1::4]
    W2 = list(v.parameters())[2::4]
    WW1 = nn.utils.parameters_to_vector(W1).view(L*(M+1),d*hidden).view(M+1,L,hidden,d)
    WW2 = nn.utils.parameters_to_vector(W2).view(L*(M+1),d*hidden).view(M+1,L,d,hidden)
    bb1 = nn.utils.parameters_to_vector(b1).view(L*(M+1),hidden).view(M+1,L,hidden)

    C = diff1_tanh(torch.tensordot(WW1,x,dims=([-1],[-1]))\
                  +torch.unsqueeze(bb1,-1).expand(M+1,L,hidden,r))#(M+1,L,hidden,r)
    output = torch.einsum('mlkh,mlhi,mlhs,m->iks', [WW2,C,WW1,Phi(t)]) #(r)
    return output


class Loss_net(nn.Module):
    def __init__(self):
        super(Loss_net, self).__init__()
        self.V = Module_v()
        self.X_solver = RK4_block_X(self.V)
        self.lnRo_solver = RK4_block_lnRo(self.V)

    def forward(self, x):

        X_all0 = x
        lnRo0 = lnrho0(x)
        #print('initial_lnRo0:',lnRo0.min(),lnRo0.size())
        t = torch.linspace(T0,T,2*N+1,device=device)
        loss1 = torch.zeros(1,device=device)
        loss_FP = torch.zeros(1,device=device)

        for n in range(0,2*N,2):

            X_all1 = self.X_solver(X_all0,t[n])
            X_all2 = self.X_solver(X_all1,t[n]+h/4)##t1,t3
            X_all3 = self.X_solver(X_all2,t[n]+h/2)
            X_all4 = self.X_solver(X_all3,t[n]+3*h/4)##t2,t4

            lnRo1 = self.lnRo_solver(lnRo0,X_all0,X_all1,X_all2,t[n])##t1,t3
            lnRo2 = self.lnRo_solver(lnRo1,X_all2,X_all3,X_all4,t[n+1])##t2,t4

            loss1 = loss1 + (torch.norm(self.V(X_all0,t[n]))**2\
                    +4*torch.norm(self.V(X_all2,t[n+1]))**2\
                    +torch.norm(self.V(X_all4,t[n+2]))**2)

            loss_FP = loss_FP + (Q(X_all0)+4*Q(X_all2)+Q(X_all4)).sum()

            X_all0 = X_all4
            lnRo0 = lnRo2


        loss1 = h/(6*r)*loss1

        loss_KL = lam_KL*torch.mean((lnRo0-lnrho1(X_all0))) ###KL

        loss_F = h/(6*r)*lam_FP*loss_FP


        loss = loss1 + loss_KL + loss_F
        return loss, loss1, loss_KL, loss_F

lossnet = Loss_net().to(device)

def rho0(x): #x(r,d) out(r)
    return lnrho0(x).exp()

def rho1(x): #x(r,d) out(r)
    return lnrho1(x).exp()

def sample_rho0(n=R):
    x = MultivariateNormal(mean0, var0).sample([n])
    return x

def sample_rho1(n=R):
    x = MultivariateNormal(mean1, var1).sample([n])
    return x


def plotkde(X2d):

    m1 = X2d[:,0]
    m2 = X2d[:,1]

    X, Y = np.mgrid[-6:6:100j, -6:6:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    fig, ax = plt.subplots()
    ax.imshow(np.rot90(Z), cmap=plt.cm.viridis,
              extent=[-6, 6, -6, 6])

    ax.set_xticks([])
    ax.set_yticks([])
    return

def plot_transport(lossnet):

    t=torch.linspace(T0, T, 2*N+1,device = device)
    X0 = sample_rho0(R)

    X_test = torch.zeros(N+1,R,d,device = device)
    RO_test = torch.zeros(N+1,R, device = device)

    X_all = torch.zeros(2*N+1,R,d,device = device)
    X_all[0] = X0

    RO_test[0] = rho0(X0)#(r)

    for n in range(0,2*N,2): # 0, 2,4,...38
        X_all[n+1] = lossnet.X_solver(X_all[n],t[n],h/2)
        X_all[n+2] = lossnet.X_solver(X_all[n+1],t[n+1],h/2)

    X_test = X_all[::2]
    X_test = X_test.cpu().detach().numpy()
    plt.figure(ite//100)

    X2d=X_test[0,:,0:2]
    plotkde(X2d)
    plt.savefig('ite'+str(ite)+'_d'+str(d)+'_push0.png',bbox_inches='tight')
    plt.show()

    X2d=X_test[int(0.2*N),:,0:2]
    plotkde(X2d)
    plt.savefig('ite'+str(ite)+'_d'+str(d)+'_push2.png',bbox_inches='tight')
    plt.show()

    X2d=X_test[int(0.5*N),:,0:2]
    plotkde(X2d)
    plt.savefig('ite'+str(ite)+'_d'+str(d)+'_push5.png',bbox_inches='tight')
    plt.show()

    X2d=X_test[int(0.8*N),:,0:2]
    plotkde(X2d)
    plt.savefig('ite'+str(ite)+'_d'+str(d)+'_push8.png',bbox_inches='tight')
    plt.show()

    X2d=X_test[N,:,0:2]
    plotkde(X2d)
    plt.savefig('ite'+str(ite)+'_d'+str(d)+'_push10.png',bbox_inches='tight')
    #plt.show()
    
    w = 6
    grid_x, grid_y = np.meshgrid(np.arange(-w+0.001, w-0.001, 0.01), np.arange(-w+0.001, w-0.001, 0.01))  
    tens_x = torch.from_numpy(grid_x)
    tens_y = torch.from_numpy(grid_y)
    size = tens_x.size()
    grid = torch.zeros(size[0],size[1],2)
    #print(grid.shape)
    grid[:,:,0]=tens_x
    grid[:,:,1]=tens_y
    grid_2d = grid.view(size[0]*size[1],2)
    
    Q_data = Q(grid_2d).view(size).numpy()
    plt.figure()
    #plt.imshow(Q_data, extent=[-w, w, -w, w])
    
    fig, ax = plt.subplots()
    ax.imshow(Q_data, cmap=plt.cm.viridis,
              extent=[-w, w, -w, w])

    ax.set_xticks([])
    ax.set_yticks([])  

    m1 = X_test[:,10:14,0]
    m2 = X_test[:,10:14,1]
    plt.plot(m1,m2,'red')
    #plt.savefig('ite'+str(ite)+'_d'+str(d)+'_push10_characteristic.png',bbox_inches='tight')

    m3 = X_test[:,2510:2514,0]
    m4 = X_test[:,2510:2514,1]
    plt.plot(m3,m4,'red')
    plt.savefig('d_'+str(d)+'_lamKL_'+str(lam_KL)+'_lamFP_'+str(lam_FP)+'_hidden_'+str(hidden)+'_ite_'+str(ite)+str(test)+'.png',bbox_inches='tight')
    plt.show()

    iX_test = torch.zeros(N+1,R,d, device=device)
    iX_all = torch.zeros(2*N+1,R,d,device=device)
    iX_all[0] = sample_rho1(R)

    for n in range(0,2*N,2): # 0, 2,4,...18
        iX_all[n+1] = lossnet.X_solver(iX_all[n],1-t[n],step=-h/2)
        iX_all[n+2] = lossnet.X_solver(iX_all[n+1],1-t[n+1],step=-h/2)

    iX_test = iX_all[::2]
    iX_test = iX_test.cpu().detach().numpy()
    X2d=iX_test[0,:,0:2]
    plotkde(X2d)
    plt.savefig('d'+str(d)+'_rho1'+'_ite'+str(ite+1)+'_r'+str(r)+'_lr'+str(lr) +'_lamKL_'+str(lam_KL)+'_lamFP_'+str(lam_FP)+'_N'+str(N)+'_M'+str(M)+'_L'+str(L)+'_loss1_'+ str(round(Loss1.item(),4)) +'_loss2_'+str(round(Loss2.item(),4))+'_loss3_'+str(round(Loss3.item(),4))+'_adam.png',bbox_inches='tight')
    plt.show()
    plt.close('all')

    return


optimizer = torch.optim.Adam(lossnet.parameters(), lr= lr )

time_begin = time.time()

for ite in range(epoch):

    print('ite:',ite)

    X = sample_rho0(r)

    Loss,Loss1,Loss2,Loss3 = lossnet(X)

    optimizer.zero_grad()
    Loss.backward()
    optimizer.step()

    if ite >0 and ite % 10 ==0:

        for p in optimizer.param_groups:
            p['lr'] = p['lr']*0.98
            q = p['lr']
            print('lr:',q)

    print('loss1:',"%.8f" %Loss1,'loss2:',"%.8f" %Loss2, 'loss3:',"%.8f" %Loss3,'loss:',"%.8f" %Loss)

    if ite % 10 == 9:
        plot_transport(lossnet)
#plot_transport(lossnet)


time_end = time.time()
time_total = time_end - time_begin
print('time:',time_total,'s')
PATH = 'lossnet_d_'+str(d)+'_lamKL_'+str(lam_KL)+'_lamFP_'+str(lam_FP)+'_hidden_'+str(hidden)+'.pt'
torch.save(lossnet,PATH)

