# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel_matmul_layer_norm(
    A, B, C,  # pointers to matrices
    mean, var,  # pointers to layer norm statistics
    W, b,       # pointers to layer norm weight/bias
    M, N, K,  # matrix dimensions
    eps,        # layer norm epsilon
    stride_am, stride_ak,  # strides for A
    stride_bk, stride_bn,  # strides for B
    stride_cm, stride_cn,  # strides for C
    stride_mean, stride_var,  # strides for mean/var
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    ram = tl.max(rm[:, None], 0)
    ran = tl.max(rn[None, :], 0)
    ram = tl.where(ram < M, ram, -1)
    ran = tl.where(ran < N, ran, -1)

    k_range = tl.arange(0, BLOCK_SIZE_K)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        ka = k + k_range
        kb = k + k_range

        a = tl.load(A + ram[:, None] * stride_am + ka[None, :] * stride_ak, mask=(ram[:, None] >= 0) & (ka[None, :] < K), other=0.0)
        b = tl.load(B + kb[:, None] * stride_bk + ran[None, :] * stride_bn, mask=(kb[:, None] >= 0) & (ran[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    c = acc.to(tl.float32)

    # Layer norm
    sum_x = tl.sum(c, axis=1)
    mean_x = sum_x / N
    sum_x_sq = tl.sum(c * c, axis=1)
    var_x = sum_x_sq / N - mean_x * mean_x
    inv_std = 1.0 / tl.sqrt(var_x + eps)
    norm_x = (c - mean_x[:, None]) * inv_std[:, None]

    # Apply weight and bias
    weight = tl.load(W + ran[None, :], mask=ran[None, :] >= 0, other=1.0)
    bias = tl.load(b + ran[None, :], mask=ran[None, :] >= 0, other=0.0)
    output = norm_x * weight[None, :] + bias[None, :]

    tl.store(C + ram[:, None] * stride_cm + ran[None, :] * stride_cn, output, mask=(ram[:, None] >= 0) & (ran[None, :] < N))
    tl.store(mean + ram, mean_x, mask=ram >= 0)
    tl.store(var + ram, var_x, mask=ram >= 0)


def fused_matmul_layer_norm(A, B, W, b, eps=1e-5):
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    mean = torch.empty((M,), device=A.device, dtype=A.dtype)
    var = torch.empty((M,), device=A.device, dtype=A.dtype)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    _kernel_matmul_layer_norm[grid](
        A, B, C, mean, var, W, b,
        M, N, K, eps,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        mean.stride(0), var.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )

    return C, mean, var