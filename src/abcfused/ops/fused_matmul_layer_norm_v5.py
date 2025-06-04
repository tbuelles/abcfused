# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel_matmul_layer_norm(
    A, B, C,  # Pointers to matrices
    mean, var,
    weight, bias,
    M, N, K,  # Matrix dimensions
    eps,
    stride_am, stride_ak,  # A strides
    stride_bk, stride_bn,  # B strides
    stride_cm, stride_cn,  # C strides
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_n = pid % num_pid_n
    pid_m = pid // num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_bn[None, :] < N) & (offs_k[:, None] + k < K), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float32)

    mean_output = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    var_output = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    # Layer Norm fused
    for m in range(BLOCK_SIZE_M):
        if offs_am[m] < M:
            row = c[m, :]
            row_mean = tl.sum(row) / N
            mean_output[m] = row_mean
            row_var = tl.sum((row - row_mean) ** 2) / N
            var_output[m] = row_var
            normalized_row = (row - row_mean) / tl.sqrt(row_var + eps)
            if weight is not None and bias is not None:
               normalized_row = normalized_row * tl.load(weight + offs_bn, mask=offs_bn < N, other=1.0) + tl.load(bias + offs_bn, mask=offs_bn < N, other=0.0)

            tl.store(C + (offs_am[m] * stride_cm + offs_bn * stride_cn), normalized_row, mask=offs_bn < N)

    tl.store(mean + offs_am, mean_output, mask=offs_am < M)
    tl.store(var + offs_am, var_output, mask=offs_am < M)


def fused_matmul_layer_norm(a, b, weight, bias, eps=1e-5, block_size_m=32, block_size_n=32, block_size_k=16):
    M, K = a.shape
    K, N = b.shape
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    assert weight.is_contiguous(), "Weight must be contiguous"
    assert bias.is_contiguous(), "Bias must be contiguous"
    assert weight.shape[0] == N, "Weight dimension mismatch"
    assert bias.shape[0] == N, "Bias dimension mismatch"

    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    mean = torch.empty((M,), device=a.device, dtype=torch.float32)
    var = torch.empty((M,), device=a.device, dtype=torch.float32)

    grid = (triton.cdiv(M, block_size_m) * triton.cdiv(N, block_size_n),)

    _kernel_matmul_layer_norm[grid](
        a, b, c, mean, var, weight, bias,
        M, N, K, eps,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=block_size_m, BLOCK_SIZE_N=block_size_n, BLOCK_SIZE_K=block_size_k
    )

    return c, mean, var