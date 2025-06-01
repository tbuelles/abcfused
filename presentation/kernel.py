# pyright: reportArgumentType=false
# pyright: reportUnreachable=false
# pyright: reportOptionalMemberAccess=false

from functools import partial
import os

from numpy import require

from abcfused.utils.dev_utils import clone_data, assert_allclose, filter_kwargs, stats_fn
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
os.environ["TRITON_DEBUG"] = "1"
# os.environ["TRITON_INTERPRET"] = "1"

import triton
import triton.language as tl
import torch

configs = [
    triton.Config(
        {},
        num_warps=nw,
        num_stages=ns,
    )
    for nw in [2, 4, 8]
    for ns in [2, 3, 4]
]

####################################################################################################
@triton.jit
def _fwd(
    x_ptr, w_ptr, scale_ptr, bias_ptr, out_ptr,
    stride_x_n, stride_x_k, stride_x_d,
    stride_w_k, stride_w_d,
    stride_out_n, stride_out_k, stride_out_d,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # program, offsets, mask
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d_offs < D
    k_offs = tl.arange(0, K)
    mask_k = k_offs < K
    mask_kd = mask_k[:, None] & mask_d[None, :]

    # load
    x = tl.load(x_ptr + pid_n * stride_x_n + k_offs[:, None] * stride_x_k + d_offs[None, :] * stride_x_d, mask=mask_kd, other=0.0)
    w = tl.load(w_ptr + k_offs[:, None] * stride_w_k + d_offs[None, :] * stride_w_d, mask=mask_kd, other=0.0)
    scale = tl.load(scale_ptr + d_offs, mask=mask_d, other=1.0)
    bias = tl.load(bias_ptr + d_offs, mask=mask_d, other=0.0)

    # compute
    tmp = tl.sum(x * w, axis=0) * scale + bias  # (BLOCK_D,)
    out = x + tl.maximum(tmp, 0.0)[None, :]  # (BLOCK_K, BLOCK_D)

    # store
    tl.store(out_ptr + pid_n * stride_out_n + k_offs[:, None] * stride_out_k + d_offs[None, :] * stride_out_d, out, mask=mask_kd)

####################################################################################################

# Autotune
_fwd_autotuned = triton.autotune(configs=configs, key=["K", "D"])(_fwd)

####################################################################################################


class _op_triton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor,
    ):
        autotune: bool = False
        BLOCK_D: int = 32
        num_warps: int = 4
        num_stages: int = 4

        N, K, D = x.shape
        out = torch.empty_like(x)

        # Choose kernel
        kernel = _fwd_autotuned if autotune else _fwd
        kwargs = {} if autotune else {"num_warps": num_warps, "num_stages": num_stages}

        # Launch forward kernel
        grid = (N, triton.cdiv(D, BLOCK_D))
        kernel[grid](
            x, w, scale, bias, out,
            x.stride(0), x.stride(1), x.stride(2),
            w.stride(0), w.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            N, K, D, BLOCK_D,
            **kwargs
        )

        # Save tensors for backward
        ctx.save_for_backward(x, w, scale, bias)
        ctx.autotune = autotune
        ctx.grid = grid
        ctx.BLOCK_D = BLOCK_D
        ctx.num_warps = num_warps
        ctx.num_stages = num_stages
        return out

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError

        x, w, scale, bias = ctx.saved_tensors
        N, K, D = x.shape
        BLOCK_D = ctx.BLOCK_D
        grid = ctx.grid

        # Allocate outputs
        dx = torch.empty_like(x)
        dw = torch.zeros_like(w)
        dscale = torch.zeros_like(scale)
        dbias = torch.zeros_like(bias)

        # Choose kernel
        kernel = _bwd_autotuned if ctx.autotune else _bwd
        kwargs = {} if ctx.autotune else {"num_warps": ctx.num_warps, "num_stages": ctx.num_stages}

        # Launch backward kernel
        kernel[grid](
            x, w, scale, bias,
            do,
            dx, dw, dscale, dbias,
            x.stride(0), x.stride(1), x.stride(2),
            w.stride(0), w.stride(1),
            do.stride(0), do.stride(1), do.stride(2),
            dx.stride(0), dx.stride(1), dx.stride(2),
            dw.stride(0), dw.stride(1),
            N, K, D, BLOCK_D,
            **kwargs
        )

        return dx, dw, dscale, dbias, None, None, None, None

def op_triton(
    x: torch.Tensor,
    w: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
):
    return _op_triton.apply(x, w, scale, bias)

if __name__ == "__main__":
    def op_torch(x, w, scale, bias):
        y = (x * w[None, :, :]).sum(dim=1)  # (N, D)
        z = torch.relu(y * scale[None, :] + bias[None, :])  # (N, D)
        return x + z[:, None, :]  # (N, K, D)
        
    def get_op_data(N=100_000, K=32, D=128, requires_grad=False):
        # Seed
        torch.manual_seed(42)

        # Representation
        x = torch.randn(N, K, D, device="cuda", requires_grad=requires_grad)

        # Learnable weights, scale, bias
        w = torch.randn(K, D, device="cuda", requires_grad=requires_grad)
        scale = torch.randn(D, device="cuda", requires_grad=requires_grad)
        bias = torch.randn(D, device="cuda", requires_grad=requires_grad)

        # Gradients
        if requires_grad:
            do = torch.randn(N, K, D, device="cuda", requires_grad=requires_grad)
        else:
            do = None

        return {"x": x, "w": w, "scale": scale, "bias": bias, "do": do}

    REQUIRES_GRAD = True

    data = get_op_data(requires_grad=REQUIRES_GRAD)

    # triton
    data_triton = filter_kwargs(op_triton, clone_data(data))
    out_triton = stats_fn(
        op_triton,
        data_triton,
        data["do"],
        label="triton",
        n_warmup=10,
        n_repeat=25,
    )

    dx_triton = data_triton["x"].grad.clone() if REQUIRES_GRAD else None
    dw_triton = data_triton["w"].grad.clone() if REQUIRES_GRAD else None
    dscale_triton = data_triton["scale"].grad.clone() if REQUIRES_GRAD else None
    dbias_triton = data_triton["bias"].grad.clone() if REQUIRES_GRAD else None
    del data_triton

    ################################################################################
    # torch
    data_torch = filter_kwargs(op_torch, clone_data(data))
    out_torch = stats_fn(
        op_torch,
        data_torch,
        data["do"],
        label="torch",
        n_warmup=10,
        n_repeat=25,
    )

    dx_torch = data_torch["x"].grad.clone() if REQUIRES_GRAD else None
    dw_torch = data_torch["w"].grad.clone() if REQUIRES_GRAD else None
    dscale_torch = data_torch["scale"].grad.clone() if REQUIRES_GRAD else None
    dbias_torch = data_torch["bias"].grad.clone() if REQUIRES_GRAD else None
    del data_torch

    assert_allclose(out_triton["o"], out_torch["o"], label="Output")
    assert_allclose(dx_triton, dx_torch, label="dx")
    assert_allclose(dw_triton, dw_torch, label="dw")
    assert_allclose(dscale_triton, dscale_torch, label="dscale")
    assert_allclose(dbias_triton, dbias_torch, label="dbias")