# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel_matmul_layer_norm(
    A, B, C,  # Pointers to matrices
    mean, var,
    W, bias,
    M, N, K,  # Matrix dimensions
    eps,
    stride_am, stride_ak,
    stride_bm, stride_bk,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bm)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_bn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = acc.to(tl.float32)
    mean_output = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    var_output = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for i in range(BLOCK_SIZE_M):
      if pid_m * BLOCK_SIZE_M + i < M:
          row = c[i, :]
          mean_val = tl.sum(row) / N
          mean_output[i] = mean_val
          var_val = tl.sum((row - mean_val) ** 2) / N
          var_output[i] = var_val
          c[i, :] = (row - mean_val) / tl.sqrt(var_val + eps)


    if W is not None and bias is not None:
      w = tl.load(W + offs_bn, mask=offs_bn < N, other=1.0)
      b = tl.load(bias + offs_bn, mask=offs_bn < N, other=0.0)
      c = c * w[None, :] + b[None, :]

    c_ptrs = C + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))
    tl.store(mean + offs_am, mean_output, mask=offs_am < M)
    tl.store(var + offs_am, var_output, mask=offs_am < M)


def fused_matmul_layer_norm(A, B, W, bias, eps):
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    mean = torch.empty((M,), device=A.device, dtype=A.dtype)
    var = torch.empty((M,), device=A.device, dtype=A.dtype)

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )

    _kernel_matmul_layer_norm[grid](
        A, B, C,
        mean, var,
        W, bias,
        M, N, K,
        eps,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return C, mean, var