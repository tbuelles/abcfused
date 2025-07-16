# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel_matmul_layer_norm(
    A, B, bias, gamma,
    C,
    M, N, K,
    stride_am, stride_ak,
    stride_bm, stride_bk,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    ram = tl.max_contiguous(tl.min(rm, M) % M)
    ran = tl.max_contiguous(tl.min(rn, N) % N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        ak = k + tl.arange(0, BLOCK_SIZE_K)
        a_mask = (ram[:, None] < M) & (ak[None, :] < K)
        b_mask = (ak[:, None] < K) & (ran[None, :] < N)

        a = tl.load(A + ram[:, None] * stride_am + ak[None, :] * stride_ak, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(B + ak[:, None] * stride_bk + ran[None, :] * stride_bm, mask=b_mask, other=0.0).to(tl.float32)

        acc += tl.dot(a, b)

    mean = tl.sum(acc, axis=1) / N
    variance = tl.sum((acc - mean[:, None])**2, axis=1) / N
    inv_std = 1.0 / tl.sqrt(variance + 1e-5)

    output = (acc - mean[:, None]) * (gamma * inv_std)[:, None] + bias[:, None]

    mask = (ram[:, None] < M) & (ran[None, :] < N)
    tl.store(C + ram[:, None] * stride_cm + ran[None, :] * stride_cn, output.to(tl.float32), mask=mask)

def matmul_layer_norm(a, b, bias, gamma):
    M, K = a.shape
    K, N = b.shape
    C = torch.empty((M, N), device=a.device, dtype=torch.float32)

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    _kernel_matmul_layer_norm[grid](
        a, b, bias, gamma,
        C,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )

    return C

fused_matmul_layer_norm = matmul_layer_norm