import torch
import torch.nn as nn
from tqdm import tqdm

class GMM(nn.Module):
    def __init__(self,means,covs,mixc):
        super().__init__()
        self.means=means # Nmix,Ndim
        self.covs=covs   # Nmix, Ndim, Ndim
        self.mixc=mixc # Nmix
        self.Nmix=self.means.shape[0]
        self.Ndim=self.means.shape[1]
        
        self.resp=None
        self.mix_resp=None

    def E_step(self,x):    
        # x: Nsamples,Ndim
        xcentered       = x[:,None,:]-self.means[None,...] # Nsamples, Nmix, Ndim
        covs_inv        = torch.linalg.inv(self.covs)
        xTQx            = torch.einsum("NMD,MDK,NMK->NM",xcentered,covs_inv,xcentered)

        mn_likelihoods  = (torch.det(covs_inv)**0.5)*torch.exp(-0.5*xTQx)/((torch.pi*2)**(self.Ndim/2)) # NM
        unnormalized_responsibilities = torch.einsum("NM,M->NM",mn_likelihoods,self.mixc) # NM
        self.resp = unnormalized_responsibilities/(unnormalized_responsibilities.sum(dim=-1,keepdim=True))# NM

        self.mix_resp = self.resp.sum(dim=0) # M

        
    def M_step(self,x):
        # x: ND
        # self.mix_resp: M
        # self.resp : NM
        
        alphas=self.resp/self.mix_resp[None,:] # NM
        #                                M_                        NM_     N_D
        self.means = torch.einsum("NM,NMD->MD",alphas,x[:,None,:]) #NMD->MD
        
                # N_D            _MD
        x_zm = x[:,None,:]-self.means[None,...] # NMD
        #                                M__                        NM_     NMD NMD
        self.covs = torch.einsum("NM,NMD,NMT->MDT",alphas,x_zm,x_zm) #
        self.mixc = self.mix_resp/x.shape[0]

    def forward(self,x,Niter):
        for n in tqdm(range(Niter)):
            self.E_step(x)
            self.M_step(x)
    