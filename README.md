## SVD-Flash

![SVD-Flash: Efficient LLM inference via SVD Compression and Tiling on AWS Trainium](./images/svd_flash.png)

## Setup Steps

1. Launch a Tranium instance using [AWS EC2](https://us-west-2.console.aws.amazon.com/ec2/home?region=us-west-2#LaunchInstances:) with the following settings:  
   i. **Name and tags**: SVD-Flash  
   ii. **Amazon Machine Image**: Deep Learning AMI Neuron (Ubuntu 22.04)  
   iii. **Instance type**: trn1.2xlarge  
   iv. **Key pair (login)**: create a new key pair  
   v. **Metadata version [under “Advanced details”]**: V2 only (otherwise, you will encounter a not authorized error)  
   vi. When connecting to these instances via SSH, use the username of *ubuntu*.

2. Activate the Neuron virtual environment to run inference by running  
   ```bash
   source /opt/aws_neuronx_venv_pytorch_2_7_nxd_inference/bin/activate

3. Download `Llama-3.2-1B` from Hugging face
    ``` 
    mkdir models

    huggingface-cli download --token  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 meta-llama/Llama-3.2-1B --local-dir ./models/llama-3.2-1b

    cd /home/ubuntu/models/llama-3.2-1b

    mv model.safetensors  model_ori.safetensors

4. Download the weights after SVD and post-training processing
   ```
   wget "https://huggingface.co/SVD-Flash/llama-3.2-1b_0.8_svd/resolve/main/llama-3.2-1b_svd_0.8_weights.safetensors?download=true" \
     -O model.safetensors   

5. Download the v0.0.1 repo
   ```
   cd ~   
   git clone -b v0.0.1 --single-branch https://github.com/dinghongsong/SVD-Flash.git


5. Testing Example (Without Tensor Parallelism): Llama inference with logit matching accuracy check using custom error tolerances
   ```
   python llama_inference.py \
    --model-type llama \
    --task-type causal-lm \
    run \
    --model-path /home/ubuntu/models/llama-3.2-1b \
    --compiled-model-path /home/ubuntu/traced_model/llama-3.2-1b \
    --torch-dtype bfloat16 \
    --batch-size 1 \
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


