# type: ignore

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def _layer_norm_matmul_kernel(
    X, W, B, Weight,
    Mean, Rstd,
    Output,
    M, N, K,
    eps,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)

    row_start = row_idx * BLOCK_SIZE_M
    col_start = col_idx * BLOCK_SIZE_N

    offsets_M = row_start + tl.arange(0, BLOCK_SIZE_M)
    offsets_N = col_start + tl.arange(0, BLOCK_SIZE_N)

    mask_M = offsets_M < M
    mask_N = offsets_N < N

    # Layer Norm part
    sum_x = tl.sum(tl.load(X + offsets_M[:, None] * K + tl.arange(0, K)[None, :], mask=mask_M[:,None]), axis=1)
    mean = sum_x / K
    tl.store(Mean + offsets_M, mean, mask=mask_M)

    sum_x2 = tl.sum(tl.load(X + offsets_M[:, None] * K + tl.arange(0, K)[None, :], mask=mask_M[:,None])**2, axis=1)
    variance = sum_x2 / K - mean**2
    rstd = 1 / tl.sqrt(variance + eps)
    tl.store(Rstd + offsets_M, rstd, mask=mask_M)

    x_norm = (tl.load(X + offsets_M[:, None] * K + tl.arange(0, K)[None, :], mask=mask_M[:,None]) - mean[:, None]) * rstd[:, None]
    x_norm = x_norm * tl.load(Weight + tl.arange(0, K)[None, :], mask=None) + tl.load(B + tl.arange(0, K)[None, :], mask=None)

    # Matmul part
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
      offsets_K = k + tl.arange(0, BLOCK_SIZE_K)
      a = tl.load(x_norm + offsets_M[:, None] * K + offsets_K[None, :], mask=mask_M[:, None] & (offsets_K[None, :] < K)).to(tl.float32)
      b = tl.load(W + offsets_K[:, None] * N + offsets_N[None, :], mask=(offsets_K[:, None] < K) & mask_N[None, :]).to(tl.float32)
      accumulator += tl.dot(a, b)

    output = accumulator.to(tl.float32)
    tl.store(Output + offsets_M[:, None] * N + offsets_N[None, :], output, mask=mask_M[:, None] & mask_N[None, :])


def fused_layer_norm_matmul(x, weight, bias, W):
    M, K = x.shape
    N = W.shape[1]
    eps = 1e-5

    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    mean = torch.empty((M,), device=x.device, dtype=x.dtype)
    rstd = torch.empty((M,), device=x.device, dtype=x.dtype)

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    _layer_norm_matmul_kernel[grid](
        x, W, bias, weight,
        mean, rstd,
        output,
        M, N, K,
        eps,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )

    return output