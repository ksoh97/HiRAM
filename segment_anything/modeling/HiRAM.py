import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .common import LayerNorm2d
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class HiRAM(nn.Module):
    def __init__(self, d_model_sam, d_model_cnn=None):
        super().__init__()
        if d_model_cnn is None: d_model_cnn = d_model_sam
        self.norm_sam = LayerNorm2d(d_model_sam)
        self.norm_cnn = LayerNorm2d(d_model_cnn)
        
        self.mug_sam = MUG(d_model_sam)
        self.mug_cnn = MUG(d_model_cnn)
        self.out_proj_sam = nn.Linear(self.mug_sam.d_inner, d_model_sam)
        self.out_proj_cnn = nn.Linear(self.mug_cnn.d_inner, d_model_cnn)

        
    def forward(self, feat_sam, feat_cnn):
        shortcut_sam = feat_sam
        shortcut_cnn = feat_cnn
        feat_sam = self.norm_sam(feat_sam)
        feat_cnn = self.norm_cnn(feat_cnn)
        
        # run MUG
        out_sam, z_sam, region_mask_sam = self.mug_sam(feat_sam)
        out_cnn, z_cnn, region_mask_cnn = self.mug_cnn(feat_cnn)
        
        # cross-branch multiplication
        out_sam = out_sam * z_cnn
        out_cnn = out_cnn * z_sam
        
        out_sam = self.out_proj_sam(out_sam.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2)
        out_cnn = self.out_proj_cnn(out_cnn.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2)
        
        out_sam = out_sam + shortcut_sam
        out_cnn = out_cnn + shortcut_cnn

        return out_sam, out_cnn, [region_mask_sam, region_mask_cnn]
                

class MUG(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, ssm_ratio=2):
        super().__init__()
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = min(math.ceil(d_model / 16), 4)
        self.d_state = d_state
        self.ssoflex = True
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.act = nn.SiLU()
        
        self.priorgenerate = RPG(d_model, d_state)
        
        self.ssm_bg   = SSM(d_model, d_state=d_state, d_conv=d_conv, ssm_ratio=ssm_ratio)
        self.ssm_bd   = SSM(d_model, d_state=d_state, d_conv=d_conv, ssm_ratio=ssm_ratio)
        self.ssm_in   = SSM(d_model, d_state=d_state, d_conv=d_conv, ssm_ratio=ssm_ratio)
        self.ssm_full = SSM(d_model, d_state=d_state, d_conv=d_conv, ssm_ratio=ssm_ratio)
        self.ssm_full_bwd = SSM(d_model, d_state=d_state, d_conv=d_conv, ssm_ratio=ssm_ratio)
        
        self.out_norm = LayerNorm2d(self.d_inner)
        self.out_act = nn.GELU() or nn.Identity()

    def region_decomposition(self, x, index):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, -1, C)
        
        x_sel = x_flat[index[:, 0], index[:, 1]]
        x_sel = x_sel.permute(1,0).unsqueeze(0).contiguous()

        return x_sel

    def region_recomposition(self, y_part, index, y_full):
        B, C, N = y_full.shape
        y_full[0, :, index[:, 1]] = y_part[0]
        return y_full

    def forward(self, hidden_states):
        xz = self.in_proj(hidden_states.permute(0, 2, 3, 1))
        x, z = xz.chunk(2, dim=-1)
        z = self.act(z)
        z = z.permute(0, 3, 1, 2).contiguous()
        x = x.permute(0, 3, 1, 2).contiguous()
        
        B, C, H, W = x.shape
        
        # index & prior generation
        priors, index, region_mask = self.priorgenerate(hidden_states)
        [prior_bg, prior_bd, prior_in] = priors
        [idx_bg, idx_bd, idx_in] = index
        
        # region decomposition
        x_bg = self.region_decomposition(x, idx_bg)    # (B, C, N_bg)
        x_bd = self.region_decomposition(x, idx_bd)    # (B, C, N_bd)
        x_in = self.region_decomposition(x, idx_in)    # (B, C, N_in)
        
        # region-adaptive SSM
        y_bg = self.ssm_bg(x_bg, prior_bg).contiguous().float()  # (B, C, N_bg)
        y_bd = self.ssm_bd(x_bd, prior_bd).contiguous().float()  # (B, C, N_bd)
        y_in = self.ssm_in(x_in, prior_in).contiguous().float()  # (B, C, N_in)
        
        # region recomposition
        y = x.new_zeros(B, C, H*W)
        self.region_recomposition(y_bg, idx_bg, y)
        self.region_recomposition(y_bd, idx_bd, y)
        self.region_recomposition(y_in, idx_in, y)
        
        # SS2D
        y_full = self.ssm_full(y, None, skip_conv=True)
        y_full_bwd = self.ssm_full_bwd(torch.flip(y, dims=[-1]), None, skip_conv=True)
        y_full = y_full + torch.flip(y_full_bwd, dims=[-1])
                
        y = rearrange(y_full.contiguous(), "b c (h w) -> b c h w", h=H, w=W)
        y = self.out_norm(y)
        y = self.out_act(y)
        
        return y.to(x.dtype), z, region_mask
        
        
class SSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, ssm_ratio=2, dt_scale=1.0, dt_init_floor=1e-4, dt_min=0.001, dt_max=0.1, ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = min(math.ceil(d_model / 16), 4)
        
        self.conv = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.act = nn.SiLU()
        
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        
        ## Initialize ##
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n", d=self.d_inner,).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
    
    def forward(self, x, prior=None, skip_conv=False):
        B, D, L = x.shape
        conv_state, ssm_state = None, None
        x = x.contiguous().float()
        
        if L == 0:
            return x
        
        if not skip_conv:
            x = self.act(self.conv(x)[..., :L].contiguous())
        
        A = -torch.exp(self.A_log.float())
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=L)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
        if prior is not None:
            C = C + prior
        
        x, dt, B, C = [x_.contiguous().float() for x_ in (x, dt, B, C)]
        
        with torch.cuda.amp.autocast(enabled=False):
            y = selective_scan_fn(x, dt, A, B, C, self.D.float(), z=None,
                delta_bias=self.dt_proj.bias.float(), delta_softplus=True, return_last_state=ssm_state is not None,)
        
        return y


class RPG(nn.Module):
    def __init__(self, d_model, d_state=16, emb_rank=64):
        super().__init__()
        self.num_tokens = 3
        
        self.shared_emb = nn.Embedding(1, emb_rank)
        self.shared_emb.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)
        
        self.region_emb_list = nn.ModuleList([
            nn.Embedding(emb_rank, d_state) for _ in range(self.num_tokens)
        ])
        for emb in self.region_emb_list:
            emb.weight.data.uniform_(-1 / emb_rank, 1 / emb_rank)
        
        self.route_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 3),
            nn.GELU(),
            nn.Linear(d_model // 3, self.num_tokens),
        )    
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Log-probability map Projection
        pred_route = self.route_proj(x.permute(0, 2, 3, 1))
        pred_route = nn.LogSoftmax(dim=-1)(pred_route)
        pred_route = pred_route.view(B, H*W, self.num_tokens)
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)
        
        
        # Region grouping
        detached_index = torch.argmax(cls_policy.detach(), dim=-1)
        idx_bg = (detached_index == 0).nonzero(as_tuple=False)
        idx_bd = (detached_index == 1).nonzero(as_tuple=False)
        idx_in = (detached_index == 2).nonzero(as_tuple=False)
        
        # Confidence-based ordering
        if idx_in.numel() > 0:
            conf_in = pred_route[idx_in[:, 0], idx_in[:, 1], 2]        # (N_in,)
            _, sort_idx = torch.sort(conf_in, descending=True)
            idx_in = idx_in[sort_idx]

        if idx_bg.numel() > 0:
            conf_bg = pred_route[idx_bg[:, 0], idx_bg[:, 1], 0]        # (N_bg,)
            _, sort_idx = torch.sort(conf_bg, descending=True)
            idx_bg = idx_bg[sort_idx]

        if idx_bd.numel() > 0:
            conf_bd = pred_route[idx_bd[:, 0], idx_bd[:, 1], 1]        # (N_bd,)
            _, sort_idx = torch.sort(conf_bd, descending=True)
            idx_bd = idx_bd[sort_idx]
                
        idx_list = [idx_bg, idx_bd, idx_in]
        
        # region-wise prior generation
        region_priors = {}
        for i, region_emb in enumerate(self.region_emb_list):
            if idx_list[i].numel() > 0:
                region_pool = self.shared_emb.weight @ region_emb.weight
                masked_weights = pred_route[idx_list[i][:, 0], idx_list[i][:, 1], i][:, None]
                region_prior = torch.matmul(masked_weights, region_pool)
                region_prior = region_prior.view(B, -1, region_prior.shape[-1]).permute(0, 2, 1)
                region_priors[i] = region_prior
            else:
                region_priors[i] = x.new_zeros((B, region_emb.weight.shape[1], 0))
           
        priors = [region_priors[0], region_priors[1], region_priors[2]]
        
        region_mask = pred_route.permute(0, 2, 1).view(B, 3, H, W)
            
        return priors, idx_list, region_mask
    
