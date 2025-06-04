# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(
    A_ptr, B_ptr, C_ptr, D_ptr,  # D is the output, pre-LayerNorm result
    weight_ptr, bias_ptr,
    MEAN_ptr, VAR_ptr, # LayerNorm intermediate results
    M, N, K,
    eps,
    NORM_M, NORM_N, # LayerNorm shape
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_weight, stride_bias,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rm = tl.max(rm, 0)
    rn = tl.max(rn, 0)
    rm = tl.min(rm, M)
    rn = tl.min(rn, N)

    k_block_offset = tl.arange(0, BLOCK_SIZE_K)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_start = k * BLOCK_SIZE_K
        a_ptrs = A_ptr + (rm[:, None] * stride_am + (k_start + k_block_offset[None, :]) * stride_ak)
        b_ptrs = B_ptr + ((k_start + k_block_offset)[:, None] * stride_bk + rn[None, :] * stride_bn)

        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & ((k_start + k_block_offset[None, :]) < K), other=0.0)
        b = tl.load(b_ptrs, mask=((k_start + k_block_offset)[:, None] < K) & (rn[None, :] < N), other=0.0)

        accumulator += tl.dot(a, b)

    accumulator = accumulator.to(tl.float32)
    
    # Store intermediate result D before LayerNorm
    D_ptrs = D_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(D_ptrs, accumulator, mask=(rm[:, None] < M) & (rn[None, :] < N))


    # LayerNorm computation
    row = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row < NORM_M

    # Calculate mean
    mean = tl.sum(accumulator, axis=1) / NORM_N
    tl.store(MEAN_ptr + row, mean, mask=row_mask)

    # Calculate variance
    variance = tl.sum((accumulator - mean[:, None])**2, axis=1) / NORM_N
    tl.store(VAR_ptr + row, variance, mask=row_mask)


    # Normalize and apply weight/bias
    norm = (accumulator - mean[:, None]) / tl.sqrt(variance[:, None] + eps)
    weight = tl.load(weight_ptr + rn, mask=rn < NORM_N, other=1.0)
    bias = tl.load(bias_ptr + rn, mask=rn < NORM_N, other=0.0)
    output = norm * weight[None, :] + bias[None, :]

    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, output, mask=(rm[:, None] < M) & (rn[None, :] < N))



def fused_matmul_layer_norm(a, b, weight, bias, eps):
    M, K = a.shape
    K, N = b.shape
    C = torch.empty((M, N), device=a.device, dtype=a.dtype)
    D = torch.empty((M, N), device=a.device, dtype=a.dtype) # Intermediate result
    
    NORM_M, NORM_N = C.shape # LayerNorm shape
    MEAN = torch.empty((NORM_M,), device=a.device, dtype=a.dtype)
    VAR = torch.empty((NORM_M,), device=a.device, dtype=a.dtype)
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GRID = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    _kernel[GRID](
        a, b, C, D, weight, bias, MEAN, VAR,
        M, N, K, eps,
        NORM_M, NORM_N,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        C.stride(0), C.stride(1),
        weight.stride(0), bias.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
    )
    return C