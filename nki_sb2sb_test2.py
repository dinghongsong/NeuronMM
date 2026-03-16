# # nki_sb2sb_test.py

# from neuronxcc import nki
# import neuronxcc.nki.nccl.collectives as ncc
# import neuronxcc.nki.isa as nisa
# import neuronxcc.nki.language as nl

# import numpy as np


# @nki.jit
# def allgather_sb2sb(
#     inp,
#     replica_groups,
#     tp_degree,
# ):
#     """
#     SBUF -> SBUF AllGather kernel
#     """

#     H, W = inp.shape
#     K = W * tp_degree
#     dtype = inp.dtype

#     # SBUF input buffer
#     in_buf = nl.ndarray((H, W), dtype=dtype, buffer=nl.sbuf)

#     # copy HBM -> SBUF
#     nisa.dma_copy(
#         dst=in_buf,
#         src=inp[0:H, 0:W],
#     )

#     # SBUF output buffer
#     out_buf = nl.ndarray((H, K), dtype=dtype, buffer=nl.sbuf)

#     # output tensor in shared HBM
#     out = nl.ndarray((H, K), dtype=dtype, buffer=nl.shared_hbm)

#     # # ---- collective ----
#     # ncc.all_gather(
#     #     [out_buf],       # dsts
#     #     [in_buf],        # srcs
#     #     replica_groups,  # replica groups
#     #     1                # gather along last dim,
        
#     # )
    
#     # ncc.all_gather(
#     #     # op=None,                     # None 表示直接拷贝，不做 reduce
#     #     srcs=[in_buf],
#     #     dsts=[out_buf],
#     #     all_gather_dim=1,            # 按最后一维 gather
#     #     replica_groups=replica_groups
#     # )
    
#     ncc.all_gather(
#         None,             # op: None 表示直接拷贝
#         [inp],            # srcs: HBM input tensor
#         [out],            # dsts: HBM output tensor  
#         replica_groups,   # replica_groups
#         1,                # all_gather_dim: 沿最后一维
#         dtype=dtype,
#     )


#     # copy SBUF -> HBM
#     nisa.dma_copy(
#         dst=out[0:H, 0:K],
#         src=out_buf
#     )

#     return out


# def main():

#     # tensor parallel size
#     tp_degree = 2

#     # local tensor shape
#     H = 128
#     W = 512

#     # create host input
#     inp = np.zeros((H, W), dtype=np.float32)

#     # fill with sample data
#     for i in range(H):
#         for j in range(W):
#             inp[i, j] = 1.0

#     # replica group
#     replica_groups = [[i for i in range(tp_degree)]]

#     # run kernel
#     out = allgather_sb2sb(
#         inp,
#         replica_groups,
#         tp_degree,
#     )

#     print("Input shape:", inp.shape)
#     print("Output shape:", out.shape)
#     print("Expected shape:", (H, W * tp_degree))


# if __name__ == "__main__":
#     main()




# nki_sb2sb_test2.py

import os
import torch
import torch.distributed as dist
from torch_neuronx import nki_jit
from neuronxcc import nki
import neuronxcc.nki.nccl.collectives as ncc
import neuronxcc.nki.language as nl
import numpy as np


@nki.jit
def allgather_sb2sb(
    inp,
    replica_groups,
    tp_degree,
):
    H, W = inp.shape
    K = W * tp_degree
    dtype = inp.dtype

    out = nl.ndarray((H, K), dtype=dtype, buffer=nl.shared_hbm)

    ncc.all_gather(
        None,
        [inp],
        [out],
        replica_groups,
        1,
        dtype=dtype,
    )

    return out


def main():
    # 初始化分布式环境
    dist.init_process_group(backend="xla")

    tp_degree = int(os.environ.get("WORLD_SIZE", 2))
    rank = dist.get_rank()

    H = 128
    W = 512

    # 每个 rank 用自己的值填充，方便验证结果
    inp = torch.full((H, W), float(rank + 1), dtype=torch.float32).to("xla")

    replica_groups = [[i for i in range(tp_degree)]]

    out = allgather_sb2sb(
        inp,
        replica_groups,
        tp_degree,
    )

    if rank == 0:
        print("Input shape: ", inp.shape)
        print("Output shape:", out.shape)
        print("Expected shape:", (H, W * tp_degree))
        print("Output[:2, :4]:", out[:2, :4])  # 验证内容

    dist.destroy_process_group()


if __name__ == "__main__":
    main()