# pl-training
## Hints
  + We need pl1.6 to support manual "allgather_bucket_size" of DeepSpeed ZeRO Stage 2 Offloading
  + Please use ``DeepSpeedCPUAdam`` when "offloading", otherwise please use ``FuseAdam``
## Environment
```shell
# Conda Env
conda create -n cuda11.1-torch1.9-pl1.6 python==3.9
conda activate cuda11.1-torch1.9-pl1.6
# Pytorch with CUDA11.1
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.1 -c pytorch -c conda-forge
# Apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# DeepSpeed
pip install deepspeed==0.6.0
# FariScale
pip install fairscale==0.4.5
# Pytorch-Lightning
pip install pytorch-lightning==1.6.1
# Transformers
pip install transformers==4.18.0
```

## Run Script
```shell
python run.py --do_train --build_from_strach --n_gpus 8
```
