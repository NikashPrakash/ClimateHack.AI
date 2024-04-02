import torch
import torch.nn as nn 
import torch.nn.functional as F

"https://ieeexplore.ieee.org/document/10221682"
"https://www.bing.com/search?q=solar%20power%20production%20spatio-temporal%20transformer%20architecture&qs=ds&form=ATCVAJ"

class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        
        
    def forward():
        pass
    
    def preprocess():
        pass
    
    def post_process():
        pass
    
# class TimeEmbedding(nn.Module):
#     def __init__(self, embedding_size, period_range=(1, 10000), device='cuda'):
#         super(TimeEmbedding, self).__init__()
#         self.embedding_size = embedding_size
#         self.period_range = period_range
#         self.device = device

#     def forward(self, timestamps):
#         positions = torch.arange(timestamps.size(1)).unsqueeze(1).to(self.device)
#         div_term = torch.exp(torch.arange(0, self.embedding_size, 2) *
#                              -(torch.log(self.period_range[1]) / self.embedding_size)).to(self.device)
#         pos_embedding = torch.zeros(timestamps.size(1), self.embedding_size).to(self.device)
#         pos_embedding[:, 0::2] = torch.sin(positions * div_term)
#         pos_embedding[:, 1::2] = torch.cos(positions * div_term)

#         return pos_embedding

class TimeEmbedding(nn.Module):
    def __init__(self, sequence_length, embedding_size):
        super(TimeEmbedding, self).__init__()
        self.embedding_size = embedding_size
        # Learnable parameters for the linear and periodic components
        self.omega = nn.Parameter(torch.randn(embedding_size - 1))
        self.phi = nn.Parameter(torch.randn(embedding_size))

        # Precomputed indices for the time steps
        self.register_buffer('time_indices', torch.arange(sequence_length).unsqueeze(1))

    def forward(self, x):
        # Allocate space for the time embedding
        time_embedding = torch.zeros((self.time_indices.size(0), self.embedding_size), device=x.device)

        # Linear component for j = 0
        time_embedding[:, 0] = self.omega[0] * self.time_indices[:, 0] + self.phi[0]

        # Periodic components for j > 0
        for j in range(self.embedding_size):
            time_embedding[:, j] = torch.sin(self.omega[j] * self.time_indices[:, 0] + self.phi[j])

        return time_embedding

class TemporalTransformer(nn.Module):
    def __init__(self, seq_len, d_model, nhead, num_classes, num_encoder_layers=3, dropout=0.1):
        super().__init__()
        self.time_embedding = TimeEmbedding(embedding_size=d_model)
        self.pos_encoder = nn.Embedding(seq_len, d_model)
        encoder_layer = TransformerEncoderWithCNNLayer(d_model, nhead, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.Linear(d_model, num_classes)
        self.AdaptAvgPool = nn.AdaptiveAvgPool1d([,d_model]) #TODO what is shape from linearLayer input 
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src += self.time_embedding(src)
        output = self.transformer_encoder(src)
        output = F.dropout(self.AdaptAvgPool(output),self.dropout)
        output = self.decoder(output)
        return output


class TransformerEncoderWithCNNLayer(nn.Module):
    def __init__(self, d_model, num_heads=3, dropout=0.1):
        super(TransformerEncoderWithCNNLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # 1D CNN sub-layer
        self.conv1d = nn.Conv1d(in_channels=d_model, out_channels=28, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        out = self.dropout(self.self_attn(src,src,src))
        norm1 = self.norm1(out + src)
        out = self.dropout(self.conv1d(norm1))
        src = norm1 + self.norm2(out)
        return src

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel_size=3, stride=1):
        super(BottleneckBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, 64, kernel_size=kernel_size, padding=1,stride=stride)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class ResidualBranch(nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel_size=3, tp=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.branch = nn.Sequential( 
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        )
        if tp == 1:
            self.branch = self.branch.insert(0,self.conv1)

    def forward(self, x):
        return self.branch(x)

class MultiBranchResidualBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, tp=-1):
        super(MultiBranchResidualBlock, self).__init__()
        if tp == -1:
            self.branches = nn.ModuleList([
                ResidualBranch(in_channels, out_channels, ks) for ks in [3, 7, 11, 19]
            ])
        else:
            self.branches = nn.ModuleList([
                ResidualBranch(in_channels, out_channels*tp, ks, tp) for ks in [3, 7, 11, 19]
            ])

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches],dim=1)


class MultiBranchResNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MultiBranchResNet, self).__init__()
        
        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_channels, 64, 3, padding=1, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.max_pool_1d = nn.MaxPool1d(kernel_size=2)
        
        self.res_block1 = MultiBranchResidualBlock()
        self.res_blocks2 = nn.Sequential([MultiBranchResidualBlock(64, 64, k) for k in range(1,25)])       
        
        self.prediction = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64*4, num_classes),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        out = self.bottleneck(x)
        pool_bottleneck = self.max_pool_1d(out)
        out = self.res_block1(out) + pool_bottleneck
        pool_middle = self.max_pool_1d(out)
        out = self.res_blocks2(out) + pool_middle #if no max pool skip conns in res_blocks2
        out = self.prediction(out)
        return out

