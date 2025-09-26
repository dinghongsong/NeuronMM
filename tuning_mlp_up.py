#!/usr/bin/env python
# filepath: /home/ubuntu/SVD-Flash/neuron_kernels/tuning_mlp_up_T.py
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np
import itertools
import sys
import math

# 从 three_mm.py 中导入需要调整参数的函数
from three_mm import fused_three_mm_XUV_transpose

def get_benchmark_latency(nki_func, kernel_args, kernel_kwargs):
    """
    一个辅助函数，用于评测 NKI kernel 并返回 P99 延迟。
    如果发生错误，则返回一个极大值，以便在比较中忽略该配置。
    """
    try:
        # 使用 nki.benchmark 装饰器来评测函数
        bench_func = nki.benchmark(warmup=5, iters=10)(nki_func)
        # 运行评测
        bench_func(*kernel_args, **kernel_kwargs)
        # 获取并返回 P99 延迟
        latency_res = bench_func.benchmark_result.nc_latency
        return latency_res.get_latency_percentile(99)
    except Exception as e:
        # 打印错误信息，便于调试
        print(f"Error with config {kernel_kwargs}: {str(e)}")
        # 如果编译或运行时出错，返回无穷大，确保这个配置不会被选为最佳
        return float('inf')

def calculate_memory_usage(M_tile_size, M_tiles_in_block, r_tile_size, r_tiles_in_block, 
                         K_tile_size, K_tiles_in_block, N_tile_size, N_tiles_in_block):
    """
    计算在 SBUF 上分配的张量占用的内存大小（字节）。
    我们需要确保所有 bfloat16 类型的张量占用的内存不超过 24MB。
    排除PSUM上的XU_psum、XU_psum_1、XUV_psum、XUV_psum_1。
    
    根据 fused_mlp_up_T 函数，SBUF上的张量包括：
    1. XU_result_buf: (r_n_blocks, r_tiles_in_block, r_tile_size, K_block_size)
    2. XU_result_buf_1: (r_n_blocks, r_tiles_in_block, r_tile_size, K_block_size)
    3. X_cache: (M_tiles_in_block, M_tile_size, K_block_size)
    4. U_cache: (M_tiles_in_block, M_tile_size, r_block_size)
    5. U_cache_1: (M_tiles_in_block, M_tile_size, r_block_size)
    6. V_cache: (r_tiles_in_block, r_tile_size, N_block_size)
    7. V_cache_1: (r_tiles_in_block, r_tile_size, N_block_size)
    8. final_result_buf: (N_tiles_in_block, N_tile_size, K_block_size)
    9. final_result_buf_1: (N_tiles_in_block, N_tile_size, K_block_size)
    """
    bytes_per_element = 2  # bfloat16 占用 2 个字节
    
    # 计算块大小
    K_block_size = K_tile_size * K_tiles_in_block
    r_block_size = r_tile_size * r_tiles_in_block
    N_block_size = N_tile_size * N_tiles_in_block
    
    # 计算各个张量的内存占用
    # 1. XU_result_buf: (r_n_blocks, r_tiles_in_block, r_tile_size, K_block_size)
    # 注意：r_n_blocks 在这里我们假设为1（最大的块）
    XU_result_buf_size = 1 * r_tiles_in_block * r_tile_size * K_block_size * bytes_per_element
    
    
    # 3. X_cache: (M_tiles_in_block, M_tile_size, K_block_size)
    X_cache_size = M_tiles_in_block * M_tile_size * K_block_size * bytes_per_element
    
    # 4. U_cache: (M_tiles_in_block, M_tile_size, r_block_size)
    U_cache_size = M_tiles_in_block * M_tile_size * r_block_size * bytes_per_element
    
    
    # 6. V_cache: (r_tiles_in_block, r_tile_size, N_block_size)
    V_cache_size = r_tiles_in_block * r_tile_size * N_block_size * bytes_per_element
    
    
    # 8. final_result_buf: (N_tiles_in_block, N_tile_size, K_block_size)
    final_result_buf_size = N_tiles_in_block * N_tile_size * K_block_size * bytes_per_element
    
    
    # 计算总内存使用量
    total_memory = (XU_result_buf_size + X_cache_size + 
                   U_cache_size + V_cache_size +
                   final_result_buf_size )
    
    return total_memory

def is_valid_configuration(K, M, r, N, M_tiles_in_block, r_tiles_in_block, K_tiles_in_block, N_tiles_in_block):
    """
    检查配置是否有效：
    1. 检查块大小是否超过输入矩阵维度（K维度除外，K可以小于block size）
    2. 检查内存使用是否超过28MB
    """
    # 设置瓦片尺寸常量（根据fused_mlp_up_T函数）
    M_tile_size = 128
    r_tile_size = 128
    K_tile_size = 512
    N_tile_size = 128
    
    # 计算块大小
    M_block_size = M_tile_size * M_tiles_in_block
    r_block_size = r_tile_size * r_tiles_in_block
    K_block_size = K_tile_size * K_tiles_in_block
    N_block_size = N_tile_size * N_tiles_in_block
    
    # 检查块大小是否超过输入矩阵维度（K维度允许小于block size）
    if M_block_size > M or r_block_size > r or N_block_size > N:
        return False
    
    # 计算内存使用量并检查是否超过限制
    memory_usage = calculate_memory_usage(
        M_tile_size, M_tiles_in_block, r_tile_size, r_tiles_in_block,
        K_tile_size, K_tiles_in_block, N_tile_size, N_tiles_in_block
    )
    
    max_memory = 28 * 1024 * 1024  # 28MB
    if memory_usage > max_memory:
        return False
    
    return True

if __name__ == "__main__":
    # 1. 定义要搜索的参数空间
    r_tiles_in_block_options = [1, 2, 4, 8, 16]
    K_tiles_in_block_options = [1, 2, 4]
    N_tiles_in_block_options = [1, 2, 4, 8, 16]
    M_tiles_in_block_options = [1, 2, 4, 8, 16]
    
    # 要测试的K值范围
    K_options = [1024, 2048, 4096, 8192]
    
    # 固定的矩阵形状
    M = 4096
    N = 14336
    r = math.ceil((M*N*0.8) / ((M + N)*128)) * 128
    
    # 遍历所有K值
    for K in K_options:
        print(f"\n----- Finding Best Config for K={K}, M={M}, r={r}, N={N} -----")
        sys.stdout.flush()
        
        # 定义输入张量的类型和形状
        X_ref = nt.tensor[[K, M], nl.bfloat16]      # (K, M)
        U_ref = nt.tensor[[M, r], nl.bfloat16]      # (M, r)
        V_ref = nt.tensor[[r, N], nl.bfloat16]      # (r, N)
        kernel_args = (X_ref, U_ref, V_ref)
        
        # 初始化用于追踪前5个最佳配置的变量
        top_configs = []  # 列表存储 (latency, config) 元组
        
        # 生成所有参数组合
        param_combinations = list(itertools.product(
            M_tiles_in_block_options,
            r_tiles_in_block_options,
            K_tiles_in_block_options,
            N_tiles_in_block_options
        ))
        
        valid_configs = 0
        # 遍历所有参数组合
        for params in param_combinations:
            M_tiles, r_tiles, K_tiles, N_tiles = params
            
            # 检查配置是否有效
            if not is_valid_configuration(K, M, r, N, M_tiles, r_tiles, K_tiles, N_tiles):
                continue
            
            valid_configs += 1
            
            kernel_kwargs = {
                "M_tiles_in_block": M_tiles,
                "r_tiles_in_block": r_tiles,
                "K_tiles_in_block": K_tiles,
                "N_tiles_in_block": N_tiles,
                "mixed_precision": True,
            }
            
            print(f"Testing config: M_tiles={M_tiles}, r_tiles={r_tiles}, K_tiles={K_tiles}, N_tiles={N_tiles}")
            sys.stdout.flush()
            
            # 运行评测并获取延迟
            latency = get_benchmark_latency(fused_three_mm_XUV_transpose, kernel_args, kernel_kwargs)
            
            # 添加到top_configs列表并保持按延迟排序
            if latency != float('inf'):
                top_configs.append((latency, kernel_kwargs.copy()))
                top_configs.sort(key=lambda x: x[0])  # 按延迟排序
                top_configs = top_configs[:5]  # 只保留前5个
                
                print(f"  Latency: {latency / 1000.0:.4f} ms")
                if len(top_configs) == 1 or latency == top_configs[0][0]:
                    print(f"  New best!")
                sys.stdout.flush()
        
        # 报告前5个最佳配置
        print("\n" + "="*60)
        print(f"Tested {valid_configs} valid configurations out of {len(param_combinations)} total combinations.")
        if top_configs:
            print(f"Top {len(top_configs)} Best Configurations for K={K}, M={M}, r={r}, N={N}:")
            
            for i, (latency, config) in enumerate(top_configs, 1):
                print(f"\n  Rank {i}:")
                print(f"    - Parameters: {config}")
                print(f"    - P99 Latency: {latency / 1000.0:.4f} ms")
                
                # 计算块大小和内存使用量
                M_tiles = config["M_tiles_in_block"]
                r_tiles = config["r_tiles_in_block"]
                K_tiles = config["K_tiles_in_block"]
                N_tiles = config["N_tiles_in_block"]
                
                M_block_size = 128 * M_tiles
                r_block_size = 128 * r_tiles
                K_block_size = 512 * K_tiles
                N_block_size = 128 * N_tiles
                
                memory_usage = calculate_memory_usage(128, M_tiles, 128, r_tiles, 512, K_tiles, 128, N_tiles)
                
                print(f"    - Block sizes: M={M_block_size}, r={r_block_size}, K={K_block_size}, N={N_block_size}")
                print(f"    - Memory usage: {memory_usage / (1024*1024):.2f} MB")
        else:
            print(f"No valid configuration found for K={K}, M={M}, r={r}, N={N}.")
        sys.stdout.flush()
        print("="*60 + "\n")
