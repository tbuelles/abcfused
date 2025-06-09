# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel_matmul_layer_norm(
    A, B, C, D,  # pointers to matrices
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_dm, stride_dn,  # strides of matrices
    M, N, K,  # dimensions of matrices
    eps,
    # shared memory for layer norm
    mean_ptr, var_ptr,
    W, B_layer_norm, # layer norm weights and biases
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # re-index
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    # create range to the block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # pointers
    A = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    D = D + (rm[:, None] * stride_dm + rn[None, :] * stride_dn)

    # masking
    mask = (rm[:, None] < M) & (rn[None, :] < N)

    # initialize acc
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # load data
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(A + k * stride_ak, mask=mask & (rk[None, :] + k < K), other=0.0)
        b = tl.load(B + k * stride_bk, mask=mask & (rk[:, None] + k < K), other=0.0)
        acc += tl.dot(a, b)

    # clamp acc to float32
    acc = acc.to(tl.float32)
    tl.store(C, acc, mask=mask)

    # Layer norm
    mean = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    var = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    sum_x = tl.sum(acc, axis=1)
    mean = sum_x / N
    mean_ptr += rm
    tl.store(mean_ptr, mean, mask=rm < M)

    sum_x_squared = tl.sum(acc * acc, axis=1)
    var = sum_x_squared / N - mean * mean
    var_ptr += rm
    tl.store(var_ptr, var, mask=rm < M)

    # Normalize
    x_norm = (acc - mean[:, None]) / tl.sqrt(var[:, None] + eps)

    # Scale and shift
    W_load = tl.load(W + rm, mask=rm < M, other=1.0)
    B_layer_norm_load = tl.load(B_layer_norm + rm, mask=rm < M, other=0.0)
    output = x_norm * W_load[:, None] + B_layer_norm_load[:, None]
    output = output.to(A.dtype.element_ty)

    tl.store(D, output, mask=mask)


def matmul_layer_norm(A, B, W, B_layer_norm, eps, transpose=False):
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device="cuda", dtype=torch.float32)
    D = torch.empty((M, N), device="cuda", dtype=A.dtype)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )
    mean = torch.empty((M,), device="cuda", dtype=torch.float32)
    var = torch.empty((M,), device="cuda", dtype=torch.float32)
    _kernel_matmul_layer_norm[grid](
        A, B, C, D,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        D.stride(0), D.stride(1),
        M, N, K,
        eps,
        mean, var,
        W, B_layer_norm,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return D

fused_matmul_layer_norm = matmul_layer_norm