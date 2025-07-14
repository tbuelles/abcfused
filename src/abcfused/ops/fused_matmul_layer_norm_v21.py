# type: ignore

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def _kernel_matmul_layer_norm(
    A, B, Mean, Var, Output, Weight, Bias,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    stride_mean,
    stride_var,
    M, N, K,
    eps,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_bn[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    output_block = accumulator.to(tl.float32)

    # LayerNorm
    mean = tl.sum(output_block, axis=1) / N
    variance = tl.sum((output_block - mean[:, None]) ** 2, axis=1) / N
    output_block = (output_block - mean[:, None]) / tl.sqrt(variance[:, None] + eps)
    output_block = output_block * tl.load(Weight + offs_bn, mask=offs_bn < N, other=0.0)[None, :] + tl.load(Bias + offs_bn, mask=offs_bn < N, other=0.0)[None, :]

    tl.store(Mean + offs_am, mean, mask=offs_am < M)
    tl.store(Var + offs_am, variance, mask=offs_am < M)
    tl.store(Output + offs_am[:, None] * stride_om + offs_bn[None, :] * stride_on, output_block.to(tl.float32), mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))


def fused_matmul_layer_norm(a, b, weight, bias, eps=1e-5):
    M, K = a.shape
    K, N = b.shape
    output = torch.empty((M, N), device=a.device, dtype=torch.float32)
    mean = torch.empty((M,), device=a.device, dtype=torch.float32)
    var = torch.empty((M,), device=a.device, dtype=torch.float32)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    _kernel_matmul_layer_norm[grid](
        a, b, mean, var, output, weight, bias,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        output.stride(0), output.stride(1),
        mean.stride(0),
        var.stride(0),
        M, N, K,
        eps,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return output