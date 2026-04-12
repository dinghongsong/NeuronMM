## NeuronMM

![SVD-Flash: Efficient LLM inference via SVD Compression and Tiling on AWS Trainium](./images/neuronmm.png)

## Setup Steps

1. Launch a Tranium instance using [AWS EC2](https://us-west-2.console.aws.amazon.com/ec2/home?region=us-west-2#LaunchInstances:) with the following settings:  
   i. **Name and tags**: SVD-Flash  
   ii. **Amazon Machine Image**: Deep Learning AMI Neuron (Ubuntu 24.04)  
   iii. **Instance type**: trn1.2xlarge  
   iv. **Key pair (login)**: create a new key pair  
   v. **Metadata version [under “Advanced details”]**: V2 only (otherwise, you will encounter a not authorized error)  
   vi. When connecting to these instances via SSH, use the username of *ubuntu*.



2. Activate the Neuron virtual environment
    ```
    echo 'source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate' | sudo tee -a ~/.bashrc

    source ~/.bashrc
    ```

3. Download `Llama-3.2-1B` from Hugging face
    ``` 
    mkdir models

    huggingface-cli download --token  <your_hf_token> meta-llama/Llama-3.2-1B --local-dir ./models/llama-3.2-1b
    ``` 

4. Download the weights after SVD and post-training processing.
   ```
    huggingface-cli download Macro2017/llama-3.2-1b_0.8_svd --local-dir ./models/llama-3.2-1b_0.8_svd
   ```


5. Testing Example of Llama inference.
   ```
   python llama_inference.py \
    --model-type llama \
    --task-type causal-lm \
    run \
    --model-path /home/ubuntu/models/llama-3.2-1b \
    --svd-model-path /home/ubuntu/models/llama-3.2-1b_0.8_svd \
    --compiled-model-path /home/ubuntu/traced_model/llama-3.2-1b \
    --torch-dtype bfloat16 \
    --batch-size 1 \
    --tp-degree 2 \
    --max-context-length 32 \
    --seq-len 64 \
    --check-accuracy-mode logit-matching \
    --divergence-difference-tol 0.005 \
    --tol-map "{5: (1e-5, 0.02)}" \
    --enable-bucketing \
    --top-k 1 \
    --pad-token-id 2 \
    --prompt "I believe the meaning of life is" \
    --prompt "The color of the sky is" \
    --compress-ratio 0.8

## Output Example
   ```
   ------------------------------------------------------------------------------------------
model:  /home/ubuntu/models/llama-3.2-1b
{
    "e2e_model": {
        "latency_ms_p50": 1299.866795539856,
        "latency_ms_p90": 1301.309323310852,
        "latency_ms_p95": 1302.1685719490051,
        "latency_ms_p99": 1302.9363083839417,
        "latency_ms_p100": 1303.1282424926758,
        "latency_ms_avg": 1300.0563144683838,
        "throughput": 49.22863670422672
    },
    "context_encoding_model": {
        "latency_ms_p50": 70.28031349182129,
        "latency_ms_p90": 70.3099250793457,
        "latency_ms_p95": 70.31856775283813,
        "latency_ms_p99": 70.3455662727356,
        "latency_ms_p100": 70.35231590270996,
        "latency_ms_avg": 70.27335166931152,
        "throughput": 455.3646473357901
    },
    "token_generation_model": {
        "latency_ms_p50": 39.081573486328125,
        "latency_ms_p90": 39.14194107055664,
        "latency_ms_p95": 39.16501998901367,
        "latency_ms_p99": 39.197025299072266,
        "latency_ms_p100": 39.25800323486328,
        "latency_ms_avg": 39.088456092342255,
        "throughput": 26.40825879839129
    }
}
------------------------------------------------------------------------------------------
model:  /home/ubuntu/models/llama-3.2-1b/svd_llama
{
    "e2e_model": {
        "latency_ms_p50": 893.8226699829102,
        "latency_ms_p90": 894.6243762969971,
        "latency_ms_p95": 894.8212623596191,
        "latency_ms_p99": 895.5112648010254,
        "latency_ms_p100": 895.683765411377,
        "latency_ms_avg": 893.8456416130066,
        "throughput": 71.60072950012662
    },
    "context_encoding_model": {
        "latency_ms_p50": 66.6283369064331,
        "latency_ms_p90": 66.76206588745117,
        "latency_ms_p95": 66.76559448242188,
        "latency_ms_p99": 66.77356719970703,
        "latency_ms_p100": 66.77556037902832,
        "latency_ms_avg": 66.64743423461914,
        "throughput": 480.1385134700057
    },
    "token_generation_model": {
        "latency_ms_p50": 26.091694831848145,
        "latency_ms_p90": 26.137852668762207,
        "latency_ms_p95": 26.164603233337402,
        "latency_ms_p99": 26.198863983154297,
        "latency_ms_p100": 26.267528533935547,
        "latency_ms_avg": 26.096261316730132,
        "throughput": 39.55578356560814
    }
}
e2e_model time (baseline):  1300.0563144683838
e2e_model time (neuronmm):  893.8456416130066
E2E Speedup:  1.4544528204247231

```
## Acknowledgements
Our code is based on [SVD-LLM](https://github.com/AIoT-MLSys-Lab/SVD-LLM) and [NXDI](https://github.com/aws-neuron/neuronx-distributed-inference).

We thank the teams for their open-source implementation.


## Citation

If you find AttnCache useful or relevant to your project and research, please kindly cite our paper:


```
@article{song2025neuronmm,
  title={NeuronMM: High-Performance Matrix Multiplication for LLM Inference on AWS Trainium},
  author={Song, Dinghong and Xu, Jierui and Yang, Weichu and Su, Pengfei and Li, Dong},
  journal={arXiv preprint arXiv:2510.25977},
  year={2025}
}
```



