# type: ignore

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _kernel(
    A,  # data ptr for input A
    B,  # data ptr for input B
    C,  # data ptr for output C
    D,  # data ptr for output D (intermediate result)
    bias, # data ptr for bias of layernorm
    weight, # data ptr for weight of layernorm
    stride_A_row,  # distance in bytes between two rows of A
    stride_A_col,  # distance in bytes between two columns of A
    stride_B_row,  # distance in bytes between two rows of B
    stride_B_col,  # distance in bytes between two columns of B
    stride_C_row,  # distance in bytes between two rows of C
    stride_C_col,  # distance in bytes between two columns of C
    stride_D_row,  # distance in bytes between two rows of D
    stride_D_col,  # distance in bytes between two columns of D
    N,  # number of rows of A/C
    K,  # number of columns of A/rows of B
    M,  # number of columns of B/C
    eps,  # layer norm epsilon
    BLOCK_SIZE_N: tl.constexpr,  # block size in the N dimension
    BLOCK_SIZE_K: tl.constexpr,  # block size in the K dimension
    BLOCK_SIZE_M: tl.constexpr,  # block size in the M dimension
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)

    pid_n = pid // num_pid_m
    pid_m = pid % num_pid_m

    block_start_n = pid_n * BLOCK_SIZE_N
    block_start_m = pid_m * BLOCK_SIZE_M

    offset_n = tl.arange(0, BLOCK_SIZE_N)
    offset_m = tl.arange(0, BLOCK_SIZE_M)
    a_offsets = (block_start_n + offset_n)[:, None] * stride_A_row + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_A_col
    b_offsets = tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_B_row + (block_start_m + offset_m)[None, :] * stride_B_col

    accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(A + a_offsets + k * stride_A_col)
        b = tl.load(B + b_offsets + k * stride_B_row)
        accumulator += tl.dot(a, b)

    # Store intermediate result
    c_offsets = (block_start_n + offset_n)[:, None] * stride_C_row + (block_start_m + offset_m)[None, :] * stride_C_col
    tl.store(C + c_offsets, accumulator)

    # Layer Norm
    d_offsets = (block_start_n + offset_n)[:, None] * stride_D_row + (block_start_m + offset_m)[None, :] * stride_D_col
    x = tl.load(C + c_offsets)

    mean = tl.sum(x, axis=1) / M
    variance = tl.sum((x - mean[:, None])**2, axis=1) / M
    x_norm = (x - mean[:, None]) / tl.sqrt(variance[:, None] + eps)
    output = x_norm * tl.load(weight + offset_n * weight.element_size())[:, None] + tl.load(bias + offset_n * bias.element_size())[:, None]
    tl.store(D + d_offsets, output)


def fused_matmul_layer_norm(A, B, bias, weight, eps):
    N, K = A.shape
    K, M = B.shape
    C = torch.empty((N, M), device=A.device, dtype=torch.float32)
    D = torch.empty((N, M), device=A.device, dtype=torch.float32)

    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    BLOCK_SIZE_M = 32

    grid = (triton.cdiv(N, BLOCK_SIZE_N) * triton.cdiv(M, BLOCK_SIZE_M),)

    _kernel[grid](
        A, B, C, D, bias, weight,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        D.stride(0), D.stride(1),
        N, K, M, eps,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_M=BLOCK_SIZE_M
    )
    return D