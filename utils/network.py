#network.py
import torch
from utils.gin_conv2 import GINConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, LeakyReLU
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from types import SimpleNamespace
import math
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

# CBAM stuff
################################################################################################################################

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=10, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) 
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=10, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

################################################################################################################################

CBAM_Dropout = 0.25
Linear_Dropout = 0.02
class LinearCBAM_ve(nn.Module):
    def __init__(self, in_features, out_features, config):
        super(LinearCBAM_ve, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.cbam = CBAM(gate_channels=out_features, reduction_ratio=config["input_data"]["reduction_ratio"])
        self.dropout1 = nn.Dropout(p=CBAM_Dropout)
        self.dropout2 = nn.Dropout(p=CBAM_Dropout)

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], x.shape[1], 1, 1)
        x = self.dropout1(x)
        x = self.cbam(x)
        x = x.reshape(x.shape[0], x.shape[1])
        x = self.dropout2(x)
        return x

class GINN(torch.nn.Module):
    def __init__(self, input_ch):
        super(GINN, self).__init__()
        
        # input_ch = 64
        output_ch = int(input_ch/2)
        self.l1 = torch.nn.Linear(input_ch, output_ch)
        torch.nn.init.xavier_uniform_(self.l1.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

        input_ch = output_ch
        output_ch = int(input_ch/2)
        self.l2 = torch.nn.Linear(input_ch, output_ch)
        torch.nn.init.xavier_uniform_(self.l2.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        return x

class GINN_ve(torch.nn.Module):
    def __init__(self, config):
        super(GINN_ve, self).__init__()
        
        input_ch = config['input_data']['observed_steps'] * 8
        output_ch = int(input_ch/2)
        self.l1 = torch.nn.Linear(input_ch, output_ch)
        torch.nn.init.xavier_uniform_(self.l1.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

        input_ch = output_ch
        output_ch = int(input_ch/2)
        self.l2 = torch.nn.Linear(input_ch, output_ch)
        torch.nn.init.xavier_uniform_(self.l2.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        
        return x

class GINN3_ve(torch.nn.Module):
    def __init__(self, config):
        super(GINN3_ve, self).__init__()
        
        input_ch = config['input_data']['observed_steps']*8
        output_ch = int(input_ch/2)
        self.forward_reshape = output_ch
        
        self.l1 = LinearCBAM_ve(input_ch, output_ch, config)
        self.dropout1 = nn.Dropout(p=Linear_Dropout)

        input_ch = output_ch
        output_ch = int(input_ch/2)
        
        self.l2 = LinearCBAM_ve(input_ch, output_ch, config)
        

    def forward(self, x):
        x = F.leaky_relu(self.l1(x))
        x = x.reshape(self.forward_reshape,-1).t()
        x = self.dropout1(x)
        x = F.leaky_relu(self.l2(x))
        
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)
       

    def forward(self, x):
        # Add positional encoding to the input tensor
        return x + self.encoding[:, :x.size(1)].detach().to(x.device)
    


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
        if self.enable_standalone_scale_spline:
            torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        ) # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        ) # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1) # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        ) # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x) # (batch, in, coeff)
        splines = splines.permute(1, 0, 2) # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0) # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff) # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        ) # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

class NetGINConv(torch.nn.Module):
    def __init__(self, num_features, output_size, config):
        super(NetGINConv, self).__init__()
        self.num_cords = 2
        self.input_steps = int(num_features/self.num_cords)
        self.config = config

        input_ch = self.num_cords
        output_ch = 64
        self.conv2Da = torch.nn.Conv2d(input_ch, output_ch, (2, 2),stride=2)
        torch.nn.init.xavier_uniform_(self.conv2Da.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

        self.cbam1 = CBAM (output_ch)
        
        input_ch = output_ch
        output_ch = output_ch*2
        self.conv2Db = torch.nn.Conv2d(input_ch, output_ch, (2, 1), stride=2)
        torch.nn.init.xavier_uniform_(self.conv2Db.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

        self.cbam2 = CBAM (output_ch)

        input_ch = output_ch
        output_ch = output_ch*2
        self.conv2Dc = torch.nn.Conv2d(input_ch, output_ch, (2, 1), stride=2)
        torch.nn.init.xavier_uniform_(self.conv2Dc.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

        self.cbam3 = CBAM (output_ch)
        feature_expansion = 2
        self.fc = torch.nn.Linear(int(num_features*2),int(num_features*2*feature_expansion))
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

        nn = GINN (self.input_steps*self.num_cords*2*feature_expansion)
        nn2 = GINN(self.input_steps*self.num_cords*2*feature_expansion)
        self.conv1 = GINConv(nn, nn2, train_eps=True)

        input_ch = output_ch
        output_ch = output_size
        self.conv2Dd = torch.nn.Conv2d(input_ch, output_ch, (1, 1))
        torch.nn.init.xavier_uniform_(self.conv2Dd.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))
        
        self.d_model = 4
        self.output_norm = torch.nn.LayerNorm(self.d_model)
        self.backbone = config['training'].get('backbone', 'transformer')
        if self.backbone == 'mamba':
            if Mamba is None:
                raise ImportError("mamba-ssm not installed but backbone='mamba' requested")
            
            self.mamba_layers = torch.nn.ModuleList([
                Mamba(
                    d_model=self.d_model,
                    d_state=config['training'].get('mamba_d_state', 16),
                    d_conv=config['training'].get('mamba_d_conv', 4),
                    expand=config['training'].get('mamba_expand', 2)
                ) for _ in range(config['training'].get('num_layers', 2))
            ])
            self.mamba_norms = torch.nn.ModuleList([
                torch.nn.LayerNorm(self.d_model) for _ in range(config['training'].get('num_layers', 2))
            ])
        else:
            self.decoder_layers = torch.nn.TransformerDecoderLayer(self.d_model, config['training']['num_heads'], config['training']['ffl'], dropout=config['training']['dropout'], batch_first=True) 
            self.decoder = torch.nn.TransformerDecoder(self.decoder_layers, num_layers=config['training']['num_layers'])
        
        self.positional_encoding = PositionalEncoding(self.d_model, 100)
        
        self.output_type = config['training'].get('output_type', 'mlp')
        if self.output_type == 'kan':
            self.kan_output = KAN([self.d_model, 2], grid_size=20, grid_range=[-2, 2])
        else:
            self.output_layer = torch.nn.Linear(self.d_model, 2)
        self.legacy_output_passthrough = False

    def create_tgt_mask(self, seq_len):
        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask.masked_fill_(mask == 0, float('-inf'))
        mask.masked_fill_(mask == 1, float('0'))
        return mask.to(self.config['training']['device'])
        
        
    def forward(self, x, x_real, edge_index, tgt, edge_weight=None):
        x1 = F.leaky_relu(self.fc(x_real))
        x1 = F.leaky_relu(self.conv1(x1, edge_index, edge_weight))
        x1 = x1.reshape(x.shape)
        x = torch.cat((x,x1),1)
        
        x = x.view(x.shape[0], self.config['input_data']['observed_steps'], -1)
        
        start_token = x[:, -1, :].unsqueeze(1)
        tgt = torch.cat((start_token, tgt), dim=1)
        tgt = self.positional_encoding(tgt)
        
        if self.backbone == 'mamba':
            combined = torch.cat((x, tgt), dim=1)
            for mamba, norm in zip(self.mamba_layers, self.mamba_norms):
                combined = combined + mamba(norm(combined))
            output = combined[:, x.size(1):, :]
        else:
            output = self.decoder.forward(tgt, x, tgt_mask=self.create_tgt_mask(tgt.size(1)))

        if self.legacy_output_passthrough:
            output = output[:, :, -2:]
        else:
            output = self.output_norm(output)
            if self.output_type == 'kan':
                output = self.kan_output(output)
            else:
                output = self.output_layer(output)
        return output
    
    def infer(self, x, x_real, edge_index, seq_len=12, edge_weight=None):
        x1 = F.leaky_relu(self.fc(x_real))
        x1 = F.leaky_relu(self.conv1(x1, edge_index, edge_weight))
        x1 = x1.reshape(x.shape)
        x = torch.cat((x,x1),1)
        
        x = x.view(x.shape[0], self.config['input_data']['observed_steps'], -1)
        
        start_token = x[:, -1, :].unsqueeze(1)
        pred = torch.empty((x.shape[0], 0, self.d_model)).to(x.device)
        for _ in range(seq_len):
            if self.backbone == 'mamba':
                combined = torch.cat((x, start_token), dim=1)
                for mamba, norm in zip(self.mamba_layers, self.mamba_norms):
                    combined = combined + mamba(norm(combined))
                out = combined
            else:
                out = self.decoder(start_token, x) 
            predt1 = out[:, -1:, :]
            start_token = torch.cat((start_token, predt1), dim=1)
            pred = torch.cat((pred, predt1), dim=1)   
        
        if self.legacy_output_passthrough:
            pred = pred[:, :, -2:]
        else:
            pred = self.output_norm(pred)
            if self.output_type == 'kan':
                pred = self.kan_output(pred)
            else:
                pred = self.output_layer(pred)
        return pred

class NetGINConv_ve(torch.nn.Module):
    def __init__(self, num_features, output_size, config):

        super(NetGINConv_ve, self).__init__()
        self.num_cords = 2
        self.input_steps = int(num_features/self.num_cords)
        self.config = config
        
        input_ch = self.num_cords
        output_ch = 64
        
        self.conv2Da = torch.nn.Conv2d(input_ch, output_ch, (2, 2),stride=3)
        torch.nn.init.xavier_uniform_(self.conv2Da.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))
        self.cbam_a = CBAM( output_ch, 16)
        
        input_ch = output_ch
        output_ch = output_ch*2
        self.conv2Db = torch.nn.Conv2d(input_ch, output_ch, (2, 1), stride=2)
        torch.nn.init.xavier_uniform_(self.conv2Db.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))
        self.cbam_b = CBAM( output_ch, 16)
        
        input_ch = output_ch
        output_ch = output_ch*2
        self.conv2Dc = torch.nn.Conv2d(input_ch, output_ch, (2, 1), stride=2)
        torch.nn.init.xavier_uniform_(self.conv2Dc.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))
        self.cbam_c = CBAM( output_ch, 16)
        
        self.fc = torch.nn.Linear(int(num_features*2),int(num_features*4))

        nn = GINN_ve(config)
        nn2 = GINN3_ve(config)
        self.conv1 = GINConv(nn, nn2, train_eps=True)

        input_ch = output_ch
        output_ch = output_size
        self.conv2Dd = torch.nn.Conv2d(input_ch, output_ch, (1, 1))
        torch.nn.init.xavier_uniform_(self.conv2Dd.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))
        
        self.d_model = 4
        self.output_norm = torch.nn.LayerNorm(self.d_model)
        self.backbone = config['training'].get('backbone', 'transformer')
        if self.backbone == 'mamba':
            if Mamba is None:
                raise ImportError("mamba-ssm not installed but backbone='mamba' requested")
            self.mamba_layers = torch.nn.ModuleList([
                Mamba(
                    d_model=self.d_model,
                    d_state=config['training'].get('mamba_d_state', 16),
                    d_conv=config['training'].get('mamba_d_conv', 4),
                    expand=config['training'].get('mamba_expand', 2)
                ) for _ in range(config['training'].get('num_layers', 2))
            ])
            self.mamba_norms = torch.nn.ModuleList([
                torch.nn.LayerNorm(self.d_model) for _ in range(config['training'].get('num_layers', 2))
            ])
        else:
            self.decoder_layers = torch.nn.TransformerDecoderLayer(self.d_model, config['training']['num_heads'], config['training']['ffl'], dropout=config['training']['dropout'], batch_first=True) 
            self.decoder = torch.nn.TransformerDecoder(self.decoder_layers, num_layers=config['training']['num_layers'])
        
        self.positional_encoding = PositionalEncoding(self.d_model, 100)
        
        self.output_type = config['training'].get('output_type', 'mlp')
        self.use_chunked = config['training'].get('use_chunked', False)
        self.chunk_size = config['training'].get('chunk_size', 5)
        
        if self.use_chunked:
            self.chunk_queries = torch.nn.Parameter(torch.randn(self.chunk_size, self.d_model) * 0.02)
            self.waypoint_embed = torch.nn.Linear(2, self.d_model)
        
        if self.output_type == 'one_shot_bezier':
            # NEW: One-shot Bezier Head
            self.output_layer = torch.nn.Linear(self.d_model, 4 * 2) # P1, P2, P3, P4
            self.one_shot_query = torch.nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
            self.pred_len = config['input_data']['prtediction_step']
        elif self.output_type == 'kan':
            self.kan_output = KAN([self.d_model, 2], grid_size=20, grid_range=[-2, 2])
        else:
            self.output_layer = torch.nn.Linear(self.d_model, 2)
        self.legacy_output_passthrough = False

    def create_tgt_mask(self, seq_len):
        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask.masked_fill_(mask == 0, float('-inf'))
        mask.masked_fill_(mask == 1, float('0'))
        return mask.to(self.config['training']['device'])

    def _infer_one_shot_bezier(self, x, start_pos):
        """One-shot Degree 4 Bezier (Predict 4 points -> Sample 25)."""
        if start_pos is None:
            start_pos = torch.zeros(x.size(0), 2, device=x.device, dtype=x.dtype)
        from utils.bezier import bezier_sample_degree4_torch
        tgt = self.one_shot_query.expand(x.size(0), -1, -1)
        tgt = self.positional_encoding(tgt)
        out = self.decoder.forward(tgt, x)
        out = self.output_norm(out)
        pred_ctrl_rel = self.output_layer(out).view(x.size(0), 4, 2)
        p0 = start_pos
        p1 = p0 + pred_ctrl_rel[:, 0]; p2 = p0 + pred_ctrl_rel[:, 1]
        p3 = p0 + pred_ctrl_rel[:, 2]; p4 = p0 + pred_ctrl_rel[:, 3]
        pred_abs = bezier_sample_degree4_torch(p0, p1, p2, p3, p4, self.pred_len)
        # Convert to step-by-step relative dx, dy
        pts_w_p0 = torch.cat([p0.unsqueeze(1), pred_abs], dim=1)
        return pts_w_p0[:, 1:] - pts_w_p0[:, :-1]

    def _infer_chunked(self, x, start_token, seq_len, start_pos):
        if start_pos is None:
            start_pos = torch.zeros(x.size(0), 2, device=x.device, dtype=x.dtype)
        num_chunks = seq_len // self.chunk_size
        all_pred = []
        chunk_start_abs = start_pos
        memory = x
        for stage in range(num_chunks):
            tgt = torch.cat([start_token, self.chunk_queries.unsqueeze(0).expand(x.size(0), -1, -1)], dim=1)
            tgt = self.positional_encoding(tgt)
            out = self.decoder.forward(tgt, memory, tgt_mask=self.create_tgt_mask(tgt.size(1)))
            out = self.output_norm(out)
            pred_chunk = self.output_layer(out[:, 1:1 + self.chunk_size, :])
            all_pred.append(pred_chunk)
            pred_abs = chunk_start_abs.unsqueeze(1) + torch.cumsum(pred_chunk, dim=1)
            chunk_start_abs = pred_abs[:, -1, :]
            prev_embed = self.waypoint_embed(pred_abs)
            memory = torch.cat([memory, prev_embed], dim=1)
            start_token = prev_embed[:, -1:, :]
        return torch.cat(all_pred, dim=1)

    def forward(self, x, x_real, edge_index, tgt, edge_weight=None, start_pos=None):
        x1 = F.leaky_relu(self.fc(x_real))
        x1 = F.leaky_relu(self.conv1(x1, edge_index, edge_weight))
        x1 = x1.reshape(x.shape)
        x = torch.cat((x,x1),1)
        x = x.view(x.shape[0], self.config['input_data']['observed_steps'], -1)
        
        if self.output_type == 'one_shot_bezier':
            return self._infer_one_shot_bezier(x, start_pos) # Training uses same path

        start_token = x[:, -1, :].unsqueeze(1)
        if self.use_chunked and tgt.dim() == 4:
            # Original Chunked Point forward (omitted for brevity, can be restored if needed)
            num_chunks = tgt.size(1)
            all_pred = []
            memory = x
            for stage in range(num_chunks):
                t = torch.cat([start_token, tgt[:, stage]], dim=1)
                t = self.positional_encoding(t)
                out = self.decoder.forward(t, memory, tgt_mask=self.create_tgt_mask(t.size(1)))
                out = self.output_norm(out)
                pred_chunk = self.output_layer(out[:, 1:1 + self.chunk_size, :])
                all_pred.append(pred_chunk)
                prev_embed = self.waypoint_embed(tgt[:, stage, :, :2])
                memory = torch.cat([memory, prev_embed], dim=1)
                start_token = prev_embed[:, -1:, :]
            return torch.cat(all_pred, dim=1)

        tgt = torch.cat((start_token, tgt), dim=1)
        tgt = self.positional_encoding(tgt)
        if self.backbone == 'mamba':
            combined = torch.cat((x, tgt), dim=1)
            for mamba, norm in zip(self.mamba_layers, self.mamba_norms):
                combined = combined + mamba(norm(combined))
            output = combined[:, x.size(1):, :]
        else:
            output = self.decoder.forward(tgt, x, tgt_mask=self.create_tgt_mask(tgt.size(1)))
        output = self.output_norm(output)
        if self.output_type == 'kan':
            return self.kan_output(output)
        return self.output_layer(output)
    
    def infer(self, x, x_real, edge_index, seq_len=12, edge_weight=None, start_pos=None):
        x1 = F.leaky_relu(self.fc(x_real))
        x1 = F.leaky_relu(self.conv1(x1, edge_index, edge_weight))
        x1 = x1.reshape(x.shape)
        x = torch.cat((x,x1),1)
        x = x.view(x.shape[0], self.config['input_data']['observed_steps'], -1)
        
        if self.output_type == 'one_shot_bezier':
            return self._infer_one_shot_bezier(x, start_pos)

        start_token = x[:, -1, :].unsqueeze(1)
        if self.use_chunked:
            return self._infer_chunked(x, start_token, seq_len, start_pos)

        pred = torch.empty((x.shape[0], 0, self.d_model)).to(x.device)
        for _ in range(seq_len):
            if self.backbone == 'mamba':
                combined = torch.cat((x, start_token), dim=1)
                for mamba, norm in zip(self.mamba_layers, self.mamba_norms):
                    combined = combined + mamba(norm(combined))
                out = combined
            else:
                out = self.decoder(start_token, x) 
            predt1 = out[:, -1:, :]
            start_token = torch.cat((start_token, predt1), dim=1)
            pred = torch.cat((pred, predt1), dim=1)   
        pred = self.output_norm(pred)
        if self.output_type == 'kan':
            return self.kan_output(pred)
        return self.output_layer(pred)
