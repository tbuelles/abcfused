# type: ignore

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def _layer_norm_matmul_kernel(
    X, W, B, Gamma, Beta,
    Y,
    MEAN, VAR,
    N_ELEMENTS,
    ROWS, COLS, K,
    eps,
    stride_xn, stride_xm,
    stride_wn, stride_wk,
    stride_bn,
    stride_gamman,
    stride_betan,
    stride_yn, stride_yk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)

    row_start = row_idx * BLOCK_SIZE_M
    col_start = col_idx * BLOCK_SIZE_N

    mean_ptr = MEAN + row_idx
    var_ptr = VAR + row_idx

    accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    for i in range(BLOCK_SIZE_M):
        row = row_start + i
        if row < ROWS:
            x_ptr = X + row * stride_xn
            for j in range(COLS):
                x = tl.load(x_ptr + j, mask=j < COLS)
                accumulator[i] += x
    
    mean = tl.sum(accumulator, axis=0) / COLS
    tl.store(mean_ptr, mean, mask=True)

    accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    for i in range(BLOCK_SIZE_M):
        row = row_start + i
        if row < ROWS:
            x_ptr = X + row * stride_xn
            for j in range(COLS):
                x = tl.load(x_ptr + j, mask=j < COLS)
                accumulator[i] += (x - mean) * (x - mean)
    
    variance = tl.sum(accumulator, axis=0) / COLS
    tl.store(var_ptr, variance, mask=True)

    a = row_start + tl.arange(0, BLOCK_SIZE_M)
    b = col_start + tl.arange(0, BLOCK_SIZE_N)
    a = tl.where(a < ROWS, a, -1)
    b = tl.where(b < K, b, -1)

    a_mask = (a >= 0)[:, None]
    b_mask = (b >= 0)[None, :]

    a = tl.max(a, 0)
    b = tl.max(b, 0)

    wa = tl.load(Gamma + a * stride_gamman, mask=a_mask).to(tl.float32)
    wb = tl.load(Beta + a * stride_betan, mask=a_mask).to(tl.float32)
    x = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for k in range(0, COLS, BLOCK_SIZE_K):
        k_block = k + tl.arange(0, BLOCK_SIZE_K)
        x_mask = (k_block[None, :] < COLS) & a_mask
        A = tl.load(X + a[:, None] * stride_xn + k_block[None, :], mask=x_mask, other=0.0).to(tl.float32)
        mean_val = tl.load(MEAN + a, mask=a_mask, other=0.0).to(tl.float32)
        var_val = tl.load(VAR + a, mask=a_mask, other=0.0).to(tl.float32)
        A_norm = (A - mean_val[:, None]) / tl.sqrt(var_val[:, None] + eps)
        A_norm = A_norm * wa[:, None] + wb[:, None]
        w = tl.load(W + k_block[None, :] * stride_wn + b[None, :], mask=x_mask & b_mask, other=0.0).to(tl.float32)
        x += tl.dot(A_norm, w)

    tl.store(Y + a[:, None] * stride_yn + b[None, :], x, mask=a_mask & b_mask)


def fused_layer_norm_matmul(X, W, B, Gamma, Beta, eps):
    ROWS, COLS = X.shape
    K = W.shape[1]
    assert X.is_contiguous()
    assert W.is_contiguous()
    assert B.is_contiguous()
    assert Gamma.is_contiguous()
    assert Beta.is_contiguous()

    Y = torch.empty((ROWS, K), device=X.device, dtype=torch.float32)
    MEAN = torch.empty((ROWS,), device=X.device, dtype=torch.float32)
    VAR = torch.empty((ROWS,), device=X.device, dtype=torch.float32)

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    grid = (triton.cdiv(ROWS, BLOCK_SIZE_M), triton.cdiv(K, BLOCK_SIZE_N))
    _layer_norm_matmul_kernel[grid](
        X, W, B, Gamma, Beta,
        Y,
        MEAN, VAR,
        X.numel(),
        ROWS, COLS, K,
        eps,
        X.stride(0), X.stride(1),
        W.stride(0), W.stride(1),
        B.stride(0),
        Gamma.stride(0),
        Beta.stride(0),
        Y.stride(0), Y.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )

    return Y