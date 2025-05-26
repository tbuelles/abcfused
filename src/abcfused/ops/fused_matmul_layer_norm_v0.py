# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel_matmul_layer_norm(
    A, B, C,  # Pointers to matrices
    mean, var,
    W, b,
    M, N, K,  # Matrix dimensions
    eps: float,
    stride_am, stride_ak,
    stride_bm, stride_bk,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.num_programs(axis=1)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    ram = tl.max(rm[:, None], 0)
    ran = tl.max(rn[None, :], 0)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        rk = k + tl.arange(0, BLOCK_SIZE_K)
        A_block_ptr = tl.make_block_ptr(
            base=A,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            offsets=(ram[:, None], rk[None, :]),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
            order=(1, 0)
        )
        B_block_ptr = tl.make_block_ptr(
            base=B,
            shape=(K, N),
            strides=(stride_bk, stride_bm),
            offsets=(rk[:, None], ran[None, :]),
            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
            order=(0, 1)
        )

        a = tl.load(A_block_ptr, mask= (ram[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(B_block_ptr, mask= (rk[:, None] < K) & (ran[None, :] < N), other=0.0)

        acc += tl.dot(a, b)

    # Layer Norm
    C_row_idx = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    C_col_idx = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    C_mask = (C_row_idx[:, None] < M) & (C_col_idx[None, :] < N)

    sum_x = tl.sum(acc, axis=1)
    mean_block = tl.sum(sum_x, axis=0) / (M * N)

    sum_x_squared = tl.sum(acc * acc, axis=1)
    var_block = tl.sum(sum_x_squared, axis=0) / (M * N) - mean_block * mean_block

    mean[pid] = mean_block
    var[pid] = var_block

    x_hat = (acc - mean_block) / tl.sqrt(var_block + eps)

    W_ptr = W + C_col_idx
    b_ptr = b + C_col_idx
    W_block = tl.load(W_ptr, mask=C_col_idx < N, other=0.0)
    b_block = tl.load(b_ptr, mask=C_col_idx < N, other=0.0)

    output = x_hat * W_block[None, :] + b_block[None, :]
    C_block_ptr = tl.make_block_ptr(
        base=C,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(C_row_idx[:, None], C_col_idx[None, :]),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0)
    )
    tl.store(C_block_ptr, output, mask=C_mask)

def fused_matmul_layer_norm(A, B, W, b, eps):
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    mean = torch.empty((1,), device=A.device, dtype=A.dtype)
    var = torch.empty((1,), device=A.device, dtype=A.dtype)
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    _kernel_matmul_layer_norm[grid](
        A, B, C, mean, var, W, b, M, N, K, eps,
        A.stride(0), A.stride(1),
        B.stride(1), B.stride(0),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return C