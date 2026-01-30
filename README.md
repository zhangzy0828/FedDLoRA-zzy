# FedDLoRA：基于双LoRA与蒸馏正则的个性化联邦微调框架
<img width="620" height="473" alt="image" src="https://github.com/user-attachments/assets/4ea8a70f-1e9c-4666-9f41-e7cd5ff413a5" />


## Requirments
该代码需要一些依赖项（Python 版本≥3.8），详情见requirements.txt文件中的说明。请按照相关库的说明进行安装或运行操作：
```bash
pip install -r requirements.txt
```
If `transformers` doesn't work, install it from source as:
```bash
pip install -U git+https://github.com/huggingface/transformers@de9255de27abfcae4a1f816b904915f0b1e23cd9
```

## Data Preparation
我们从 [FLAN](https://github.com/google-research/FLAN) 中构建了联邦数据集。每个数据集包含八个任务，每个客户端都被分配了特定的一个任务。
  
## Running
我们代码的架构基于 [FederatedGPT](https://github.com/JayZhang42/FederatedGPT-Shepherd). 具体而言，我们对 Hugging Face 的 [PEFT](https://github.com/huggingface/peft) 进行了调整，以实现全局和局部适配器的联合优化。

用法示例:
```bash
python main.py --global_model 'meta-llama/Llama-2-7b-hf'\
      --data_path  "./data/dataset1" \
      --output_dir  './lora-7b/'\
      --num_communication_rounds 20 \
      --num_clients  8 \
      --prompt_template_name 'alpaca_short' \
      --client_selection_frac 1 \
```

我们还可以调整一些超参数：
```bash
python main.py --global_model 'meta-llama/Llama-2-7b-hf'\
      --data_path  "./data/dataset1" \
      --output_dir  './lora-7b/'\
      --num_communication_rounds 20 \
      --num_clients  8 \
      --client_selection_frac 1 \
      --local_num_epochs  10 \
      --local_batch_size  64 \
      --local_micro_batch_size 32 \
      --local_learning_rate 0.0003 \
      --lora_r 8 \
      --lora_target_modules='[q_proj,v_proj]' \
      --kd_alpha 0.1
      --neftune_noise_alpha 5
```

## Inference 

```bash
python infer.py \
      --load_8bit \
      --base_model 'meta-llama/Llama-2-7b-hf' \
      --lora_weights_path /output/path/to/lora_weights  \
      --lora_config_path /output/path/to/lora_config   \
      --prompt_template 'alpaca_short' \
      --output_file 'out/result.jsonl' \
      --test_file './data/FLAN1/test/test_client_{client_id}.jsonl'
```

