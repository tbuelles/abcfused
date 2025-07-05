# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel_matmul_layer_norm(
    A, B, C, D,  # pointers
    M, N, K,  # shapes
    eps,  # layer_norm parameter
    stride_am, stride_ak,  # strides A
    stride_bk, stride_bn,  # strides B
    stride_cm, stride_cn,  # strides C
    stride_dm, stride_dn,  # strides D
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    ram = tl.max_contiguous(tl.min(rm, M) - rm, BLOCK_SIZE_M)
    ran = tl.max_contiguous(tl.min(rn, N) - rn, BLOCK_SIZE_N)

    A = A + (rm[:, None] * stride_am + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_ak)
    B = B + (tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_bk + rn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        ak = A + k * stride_ak
        bk = B + k * stride_bk
        a = tl.load(ak, mask= (ram[:, None] < M) & (tl.arange(0, BLOCK_SIZE_K)[None, :] < K-k), other=0.0).to(tl.float32)
        b = tl.load(bk, mask= (tl.arange(0, BLOCK_SIZE_K)[:, None] < K-k) & (ran[None, :] < N), other=0.0).to(tl.float32)
        acc += tl.dot(a, b)

    mean = tl.sum(acc, axis=1) / N
    var = tl.sum((acc - mean[:, None])**2, axis=1) / N
    invstd = 1.0 / tl.sqrt(var + eps)

    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    D = D + (rm[:, None] * stride_dm + rn[None, :] * stride_dn)

    output = (acc - mean[:, None]) * invstd[:, None]
    tl.store(C, output.to(tl.float32), mask= (ram[:, None] < M) & (ran[None, :] < N))
    tl.store(D, mean.to(tl.float32), mask= ram < M) # Storing mean for verification purpose

def fused_matmul_layer_norm(a, b, eps=1e-5):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    d = torch.empty((M,), device=a.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    _kernel_matmul_layer_norm[grid](
        a, b, c, d,
        M, N, K,
        eps,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        d.stride(0), 1,
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32
    )
    return c, d