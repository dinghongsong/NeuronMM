# Quick Start

## Setup Steps

1. Create a Trainium instance using AWS EC2 with the following settings:
    1. **AMI:** **Deep Learning AMI Neuron (Ubuntu 24.04)**
    2. **Instance type:** trn1.2xlarge
    3. **Key pair (login):** create a new key pair

2. Activate the Neuron virtual environment

    ```

    echo 'source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate' | sudo tee -a ~/.bashrc

    source ~/.bashrc

    ```

3. Download the [Llama3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) model 

```
   # full model
    huggingface-cli download --token  <your_token> meta-llama/Llama-3.2-1B --local-dir /home/ubuntu/NeuronMM/models/llama-3.2-1b/

    huggingface-cli download --token hf_<your token>  meta-llama/Llama-2-7b-hf --local-dir /home/ubuntu/NeuronMM/models/llama2-7b/

    LOtCtkFzOLFrdTfOTJAzJArYNtRdSCCqhp

    huggingface-cli download --token hf_<your token>  meta-llama/Llama-3.1-8B --local-dir /home/ubuntu/NeuronMM/models/llama3-8b/

    huggingface-cli download --token hf_<your token>  Qwen/Qwen2.5-7B-Instruct --local-dir /home/ubuntu/NeuronMM/models/qwen2.5-7b/

    huggingface-cli download --token hf_<your token>  Qwen/Qwen3-8B --local-dir /home/ubuntu/NeuronMM/models/qwen3-8b/

   huggingface-cli download Macro2017/llama-3.2-1b_0.8_svd --local-dir /home/ubuntu/NeuronMM/models/llama-3.2-1b_0.8_svd

   huggingface-cli download Macro2017/llama2-7b_0.8_svd --local-dir /home/ubuntu/NeuronMM/models/llama2-7b_0.8_svd

   huggingface-cli download Macro2017/llama3-8b_0.8_svd --local-dir /home/ubuntu/NeuronMM/models/llama-3-8b_0.8_svd
```

4. Download repo:

```
git clone -b trn2 --single-branch https://github.com/dinghongsong/NeuronMM.git

cd NeuronMM

rm -rf /tmp/nxd_model/
rm -rf /var/tmp/neuron-compile-cache/

# run svd

python main.py --enable-nki --mode evaluate_all --seq-len 2048 --context-encoding-buckets 1024 --tp-degree 2 --model-path /home/ubuntu/models/llama-3.2-1b_0.8_svd/


# run baseline

python main.py --enable-nki --mode evaluate_all --seq-len 2048 --context-encoding-buckets 1024 --tp-degree 2

python main.py --enable-nki --mode evaluate_all --seq-len 256 --context-encoding-buckets 128 --tp-degree 2


## 

python main.py   --enable-nki   --mode evaluate_all   --seq-len 32   --context-encoding-buckets 16   --tp-degree 2   --svd-model-path /home/ubuntu/NeuronMM/models/llama2-7b_0.8_svd/   --model-path /home/ubuntu/NeuronMM/models/llama2-7b --prompt ["I believe the meaning of life is"] > llama2_7b.log


python main.py   --enable-nki   --mode evaluate_all   --seq-len 2048   --context-encoding-buckets 1024   --tp-degree 2   --svd-model-path /home/ubuntu/NeuronMM/models/llama2-7b_0.8_svd/   --model-path /home/ubuntu/NeuronMM/models/llama2-7b  > llama2_7b_2k.log


python main.py   --enable-nki   --mode evaluate_all   --seq-len 2048   --context-encoding-buckets 1024   --token-generation-buckets 1 --tp-degree 2   --svd-model-path /home/ubuntu/NeuronMM/models/llama3-8b_0.8_svd/   --model-path /home/ubuntu/NeuronMM/models/llama3-8b  > llama3_8b_2k.log

```



5. InfluxDB2 Installation 




```
# install influxdb2
# Download the InfluxDB CLI client
wget https://dl.influxdata.com/influxdb/releases/influxdb2-client-2.7.5-linux-amd64.tar.gz
tar xvfz influxdb2-client-2.7.5-linux-amd64.tar.gz
sudo mv influx /usr/local/bin/

# Import the GPG signing key from keyserver
sudo gpg --keyserver keyserver.ubuntu.com --recv-keys DA61C26A0585BD3B
sudo gpg --export DA61C26A0585BD3B | sudo tee /etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg > /dev/null

# Add the InfluxData repository to apt sources
echo 'deb [signed-by=/etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg] https://repos.influxdata.com/debian stable main' \
  | sudo tee /etc/apt/sources.list.d/influxdata.list

# Update package list and install influxdb2
sudo apt-get update && sudo apt-get install -y influxdb2

# Start InfluxDB and enable it on boot
sudo systemctl start influxdb
sudo systemctl enable influxdb

# Verify the service is running
sudo systemctl status influxdb

# Confirm port 8086 is listening
ss -tlnp | grep 8086

# Create admin user, organization, and default bucket
influx setup \
  --username admin \
  --password admin123 \
  --org myorg \
  --bucket mybucket \
  --force
```


6. profile
```
ls /tmp/nxd_model/


neuron-profile capture -n  /tmp/nxd_model/context_encoding_model/_tp0_bk0/graph.neff -s profile.ntff --profile-nth-exec=2  

neuron-profile view -n  /tmp/nxd_model/context_encoding_model/_tp0_bk0/graph.neff -s profile_rank_0_exec_2.ntff 


```


## Acknowledgements
Our code is based on [SVD-LLM](https://github.com/AIoT-MLSys-Lab/SVD-LLM) and [NXDI](https://github.com/aws-neuron/neuronx-distributed-inference).

We thank the teams for their open-source implementation.
