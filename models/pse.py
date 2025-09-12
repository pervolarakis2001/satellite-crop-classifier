import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class PixelSetEncoder(nn.Module):
    def __init__(self, input_dim, mlp1=[10, 32, 64], pooling='mean_std', mlp2=[64, 128], with_extra=True, ):
        super(PixelSetEncoder, self).__init__()

        self.input_dim = input_dim
        self.mlp1_dim = copy.deepcopy(mlp1)
        self.mlp2_dim = copy.deepcopy(mlp2)
        self.pooling = pooling

        self.with_extra = with_extra
      
        #validate mpl1 and mlp2 dimensions 
        assert self.mlp1_dim[0] == self.input_dim, \
            f"mlp1 first dim ({self.mlp1_dim[0]}) != input_dim ({self.input_dim})"
        inter_dim = self.mlp1_dim[-1] * len(self.pooling.split('_'))
        if self.with_extra:
            inter_dim += self.extra_size
        assert self.mlp2_dim[0] == inter_dim, \
            f"mlp2 first dim ({self.mlp2_dim[0]}) != pooled feature dim ({inter_dim})"
 
        # Build mlp1: per-pixel on channel dim
        layers1 = []
        for i in range(len(self.mlp1_dim)-1):
            layers1.append(nn.Linear(self.mlp1_dim[i], self.mlp1_dim[i+1]))
            layers1.append(nn.BatchNorm1d(self.mlp1_dim[i+1]))
            layers1.append(nn.ReLU())
        self.mlp1 = nn.Sequential(*layers1)

        # Build mlp2: on pooled features
        layers2 = []
        for i in range(len(self.mlp2_dim) - 1):
            layers2.append(nn.Linear(self.mlp2_dim[i], self.mlp2_dim[i + 1]))
            layers2.append(nn.BatchNorm1d(self.mlp2_dim[i + 1]))
            if i < len(self.mlp2_dim) - 2:
                layers2.append(nn.ReLU())
        self.mlp2 = nn.Sequential(*layers2)
         # final output_dim
        self.output_dim = self.mlp2_dim[-1] if self.mlp2_dim else inter_dim

    def forward(self, batch):
        """
        Accepts either:
         - batch dict: { 'pixels':Tensor[B,S,C,N], 'mask':Tensor[B,S,N], 'extra':Optional[B,S,extra_size] }
         - batch tuple: (pixels, mask, extra) or ((pixels, mask), extra)
        Returns:
          - out: Tensor [B, S, output_dim]
        """
        # Unpack if tuple input
        if not isinstance(batch, dict):
            # previous convention: input = ((pixels, mask), extra)
            if isinstance(batch[0], tuple) and len(batch[0]) == 2:
                (pixels, mask), extra = batch
            else:
                pixels, mask, extra = batch
        else:
            pixels = batch['pixels']
            mask = batch['mask']
            extra = batch.get('extra', None)

        # Ensure pixels are [B, S, C, N]
        if pixels.dim() == 4 and pixels.shape[3] == self.input_dim and pixels.shape[2] != self.input_dim:
            pixels = pixels.permute(0, 1, 3, 2)

        B, S, C, N = pixels.shape
        # Flatten batch and sequence dims for per-pixel MLP
        x = pixels.view(B * S, C, N)   # [B*S, C, N]
        m = mask.view(B * S, N)        # [B*S, N]

        # Apply mlp1 to each pixel's feature vector
        x = x.permute(0, 2, 1).reshape(-1, C)  # [B*S*N, C]
        x = self.mlp1(x)                       # [B*S*N, hidden]
        hidden_dim = x.shape[-1]
        x = x.view(B * S, N, hidden_dim).permute(0, 2, 1)  # [B*S, hidden, N]

        # Pool across pixels
        pooled = torch.cat([pooling_methods[p](x, m) for p in self.pooling.split('_')], dim=1)

        # Concatenate extra features if provided
        if self.with_extra and extra is not None:
            e = extra.view(B * S, -1)        # [B*S, extra_size]
            pooled = torch.cat([pooled, e], dim=1)  # [B*S, pooled_dim]

        # Apply mlp2
        out = self.mlp2(pooled)            # [B*S, output_dim]
        out = out.view(B, S, -1)           # [B, S, output_dim]
        return out




class linlayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(linlayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lin = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, input):
        out = input.permute((0, 2, 1))
        out = self.lin(out)
        out = out.permute((0, 2, 1))
        out = self.bn(out)
        out = F.relu(out)
        return out


def masked_mean(x, mask):
    out = x.permute((1, 0, 2))
    out = out * mask
    out = out.sum(dim=-1) / mask.sum(dim=-1)
    out = out.permute((1, 0))
    return out

def masked_std(x, mask):
    m = masked_mean(x, mask)
    out = x.permute((2, 0, 1))
    out = out - m
    out = out.permute((2, 1, 0))
    out = out * mask
    d = mask.sum(dim=-1)
    d[d == 1] = 2
    out = (out ** 2).sum(dim=-1) / (d - 1)
    out = torch.sqrt(out + 10e-32)
    out = out.permute(1, 0)
    return out

def maximum(x, mask):
    return x.max(dim=-1)[0].squeeze()

def minimum(x, mask):
    return x.min(dim=-1)[0].squeeze()

pooling_methods = {
    'mean': masked_mean,
    'std': masked_std,
    'max': maximum,
    'min': minimum
}
