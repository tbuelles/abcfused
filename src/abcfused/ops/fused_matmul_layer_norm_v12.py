# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel_matmul_layer_norm(
    A, B, C,  # Pointers to matrices
    mean, var, # Pointers to layer norm output
    W, B_LN, # Pointers to layer norm parameters
    M, N, K,  # Matrix dimensions
    eps: float, # layer norm epsilon
    stride_am, stride_ak,  # Stride of matrix A
    stride_bk, stride_bn,  # Stride of matrix B
    stride_cm, stride_cn,  # Stride of matrix C
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rm = tl.max(rm, 0)
    rn = tl.max(rn, 0)
    rm = tl.minimum(rm, M)
    rn = tl.minimum(rn, N)

    k_block_offset = tl.arange(0, BLOCK_SIZE_K)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        k_start = k + k_block_offset
        a = tl.load(A + (rm[:, None] * stride_am + k_start[None, :] * stride_ak), mask=(rm[:, None] < M) & (k_start[None, :] < K), other=0.0)
        b = tl.load(B + (k_start[:, None] * stride_bk + rn[None, :] * stride_bn), mask=(k_start[:, None] < K) & (rn[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)

    c = accumulator.to(tl.float32)

    tl.store(C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn), c, mask=(rm[:, None] < M) & (rn[None, :] < N))

    # Layer Norm
    c_mean = tl.sum(c, 1) / N
    c_var = tl.sum((c - c_mean[:,None])**2, 1) / N

    x_norm = (c - c_mean[:, None]) / tl.sqrt(c_var[:, None] + eps)
    output = x_norm * tl.load(W, rm, mask = rm < M, other = 0) + tl.load(B_LN, rm, mask = rm < M, other = 0)
    tl.store(mean + rm, c_mean, mask = rm < M)
    tl.store(var + rm, c_var, mask = rm < M)
    tl.store(C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn), output, mask=(rm[:, None] < M) & (rn[None, :] < N))



def fused_matmul_layer_norm(A, B, W, B_LN, eps):
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device="cuda", dtype=torch.float32)
    mean = torch.empty((M,), device="cuda", dtype=torch.float32)
    var = torch.empty((M,), device="cuda", dtype=torch.float32)

    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )
    _kernel_matmul_layer_norm[grid](
        A, B, C, mean, var, W, B_LN,
        M, N, K, eps,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return C, mean, var