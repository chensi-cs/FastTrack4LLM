import torch 



class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        max_len: int = 1024
        )->None:
        super().__init__()
        pe=torch.zeros(max_len,d_model)
        # print("pe shape:",pe.shape)
        position=torch.arange(0,max_len).unsqueeze(1).float()
        # print("position shape:",position.shape)
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-np.log(1000.0)/d_model))
        # print("div_term shape:",div_term.shape)
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        pe.unsqueeze(0) #[1, max_len, d_model]
        # print("pe shape:",pe.shape)
        pe=pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)  # 注册为buffer，不会被梯度更新,'pe' 是这个张量的名称

    def forward(self,x):
        """
            x: Tensor shape [batch_size, seq_len, d_model]
        """
        return self.pe[:,:x.size(1),:]
    
 
 
def Sinusoidal_position_embed(config):

def Position_embed(config):

