# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel_matmul_layer_norm(
    A, B, bias, gamma,
    C,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    M, N, K,
    eps,
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
    rm_mask = rm < M
    rn_mask = rn < N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        rk = k + tl.arange(0, BLOCK_SIZE_K)
        rak = tl.max(rk[None, :], 0)
        rbk = tl.max(rk[:, None], 0)

        a = tl.load(A + (ram[:, None] * stride_am + rak[None, :] * stride_ak), mask=(rm_mask[:, None] & (rk < K)[None, :]), other=0.0).to(tl.float32)
        b = tl.load(B + (rbk[:, None] * stride_bk + ran[None, :] * stride_bn), mask=((rk < K)[:, None] & rn_mask[None, :]), other=0.0).to(tl.float32)

        accumulator += tl.dot(a, b)

    accumulator = accumulator.to(tl.float32)
    c = accumulator + tl.load(bias + rn, mask=rn_mask, other=0.0).to(tl.float32)

    mean = tl.sum(c, axis=1) / N
    variance = tl.sum((c - mean[:, None])**2, axis=1) / N
    c = (c - mean[:, None]) / tl.sqrt(variance[:, None] + eps)
    c = c * tl.load(gamma + rn, mask=rn_mask, other=1.0).to(tl.float32)

    tl.store(C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn), c, mask=rm_mask[:, None] & rn_mask[None, :])


def fused_matmul_layer_norm(a, b, bias, gamma, eps=1e-5):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    _kernel_matmul_layer_norm[grid](
        a, b, bias, gamma,
        c,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        M, N, K,
        eps,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return c