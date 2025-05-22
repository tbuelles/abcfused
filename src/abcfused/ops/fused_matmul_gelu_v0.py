# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_n = pid % num_pid_n
    pid_m = pid // num_pid_n

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    ram = tl.max(rm[:, None], 0)
    ran = tl.max(rn[None, :], 0)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        ak = k + tl.arange(0, BLOCK_SIZE_K)
        a = tl.load(A + ram * stride_am + ak[None, :] * stride_ak, mask=(ram[:, None] < M) & (ak[None, :] < K), other=0.0)
        b = tl.load(B + ak[:, None] * stride_bk + ran[None, :] * stride_bn, mask=(ak[:, None] < K) & (ran[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
    
    # GELU approximation
    x = accumulator
    gelu_approx = 0.5 * x * (1.0 + tl.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    
    c = gelu_approx

    tl.store(C + rm[:, None] * stride_cm + rn[None, :] * stride_cn, c, mask=(rm[:, None] < M) & (rn[None, :] < N))

def fused_matmul_gelu(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )
    _kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return c