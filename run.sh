# python llama_inference.py  --model-type llama  --task-type causal-lm  run  --model-path /home/ubuntu/models/llama-3.2-1b   --compiled-model-path /home/ubuntu/traced_model/llama-3.2-1b   --torch-dtype bfloat16  --batch-size 1  --max-context-length 128  --seq-len 256  --check-accuracy-mode logit-matching  --divergence-difference-tol 0.005  --tol-map "{5: (1e-5, 0.02)}"  --enable-bucketing  --top-k 1  --pad-token-id 2  --prompt "I believe the meaning of life is"  --prompt "The color of the sky is"  --compress-ratio 0.8
# python llama_inference.py  --model-type llama  --task-type causal-lm  run  --model-path /home/ubuntu/models/Llama-3.2-3B   --compiled-model-path /home/ubuntu/traced_model/Llama-3.2-3B   --torch-dtype bfloat16  --batch-size 1  --max-context-length 128  --seq-len 256  --check-accuracy-mode logit-matching  --divergence-difference-tol 0.005  --tol-map "{5: (1e-5, 0.02)}"  --enable-bucketing  --top-k 1  --pad-token-id 2  --prompt "I believe the meaning of life is"  --prompt "The color of the sky is"  --compress-ratio 0.8
# python llama_inference.py  --model-type llama  --task-type causal-lm  run  --model-path /home/ubuntu/models/Llama-3.1-8B   --compiled-model-path /home/ubuntu/traced_model/Llama-3.1-8B   --torch-dtype bfloat16  --batch-size 1  --max-context-length 128  --seq-len 256  --check-accuracy-mode logit-matching  --divergence-difference-tol 0.005  --tol-map "{5: (1e-5, 0.02)}"  --enable-bucketing  --top-k 1  --pad-token-id 2  --prompt "I believe the meaning of life is"  --prompt "The color of the sky is"  --compress-ratio 0.8

python qwen_inference.py  --model-type llama  --task-type causal-lm  run  --model-path /home/ubuntu/models/Qwen3-4B   --compiled-model-path /home/ubuntu/traced_model/Qwen3-4B   --torch-dtype bfloat16  --batch-size 1  --max-context-length 128  --seq-len 256  --check-accuracy-mode logit-matching  --divergence-difference-tol 0.005  --tol-map "{5: (1e-5, 0.02)}"  --enable-bucketing  --top-k 1  --pad-token-id 2  --prompt "I believe the meaning of life is"  --prompt "The color of the sky is"  --compress-ratio 0.8
python qwen_inference.py  --model-type llama  --task-type causal-lm  run  --model-path /home/ubuntu/models/Qwen3-8B   --compiled-model-path /home/ubuntu/traced_model/Qwen3-8B   --torch-dtype bfloat16  --batch-size 1  --max-context-length 128  --seq-len 256  --check-accuracy-mode logit-matching  --divergence-difference-tol 0.005  --tol-map "{5: (1e-5, 0.02)}"  --enable-bucketing  --top-k 1  --pad-token-id 2  --prompt "I believe the meaning of life is"  --prompt "The color of the sky is"  --compress-ratio 0.8

# #!/bin/bash

# # Define different model paths
# models=(
#     "llama-3.2-1b"
#     "Llama-3.2-3B"
#     "Llama-3.1-8B"
# )

# # Define different compression ratios
# ratios=(0.95 0.9 0.85 0.8)

# # Run in loops
# for model in "${models[@]}"; do
#   for ratio in "${ratios[@]}"; do
#     echo ">>> Running model $model with compress-ratio=$ratio"
#     python llama_inference.py \
#       --model-type llama \
#       --task-type causal-lm \
#       run \
#       --model-path /home/ubuntu/models/$model \
#       --compiled-model-path /home/ubuntu/traced_model/$model \
#       --torch-dtype bfloat16 \
#       --batch-size 1 \
#       --max-context-length 128 \
#       --seq-len 256 \
#       --check-accuracy-mode logit-matching \
#       --divergence-difference-tol 0.005 \
#       --tol-map "{5: (1e-5, 0.02)}" \
#       --enable-bucketing \
#       --top-k 1 \
#       --pad-token-id 2 \
#       --prompt "I believe the meaning of life is" \
#       --prompt "The color of the sky is" \
#       --compress-ratio $ratio
#   done
# done


# # python llama_inference.py  --model-type llama  --task-type causal-lm  run  --model-path /home/ubuntu/models/llama-3.2-1b   --compiled-model-path /home/ubuntu/traced_model/llama-3.2-1b   --torch-dtype bfloat16  --batch-size 1  --max-context-length 256  --seq-len 512  --check-accuracy-mode logit-matching  --divergence-difference-tol 0.005  --tol-map "{5: (1e-5, 0.02)}"  --enable-bucketing  --top-k 1  --pad-token-id 2  --prompt "I believe the meaning of life is"  --prompt "The color of the sky is"  --compress-ratio 0.8
# # python llama_inference.py  --model-type llama  --task-type causal-lm  run  --model-path /home/ubuntu/models/Llama-3.2-3B   --compiled-model-path /home/ubuntu/traced_model/Llama-3.2-3B   --torch-dtype bfloat16  --batch-size 1  --max-context-length 256  --seq-len 512  --check-accuracy-mode logit-matching  --divergence-difference-tol 0.005  --tol-map "{5: (1e-5, 0.02)}"  --enable-bucketing  --top-k 1  --pad-token-id 2  --prompt "I believe the meaning of life is"  --prompt "The color of the sky is"  --compress-ratio 0.8
# # python llama_inference.py  --model-type llama  --task-type causal-lm  run  --model-path /home/ubuntu/models/Llama-3.1-8B   --compiled-model-path /home/ubuntu/traced_model/Llama-3.1-8B   --torch-dtype bfloat16  --batch-size 1  --max-context-length 256  --seq-len 512  --check-accuracy-mode logit-matching  --divergence-difference-tol 0.005  --tol-map "{5: (1e-5, 0.02)}"  --enable-bucketing  --top-k 1  --pad-token-id 2  --prompt "I believe the meaning of life is"  --prompt "The color of the sky is"  --compress-ratio 0.8