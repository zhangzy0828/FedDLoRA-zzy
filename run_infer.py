import os
import fire
from infer import main 

def run_all():
    base_dir = "/2014110093/zhangzy/FedDPA"
    lora_base_path = os.path.join(base_dir, "output/train/KD/8/19")
    
    for client_id in range(8):  # 0 to 7
        print(f"\n Running inference for client {client_id}...\n")
        
        local_model_path = os.path.join(lora_base_path, f"local_output_{client_id}/local/pytorch_model.bin")
        output_file = os.path.join(base_dir, f"output/infer/KD/results_{client_id}.jsonl")
        test_file = os.path.join(base_dir, f"data/FLAN1/test/test_client_{client_id}.jsonl")

        # 检查文件是否存在
        if not os.path.exists(local_model_path):
            print(f" Warning: local model not found for client {client_id}, skipping.")
            continue
        if not os.path.exists(test_file):
            print(f" Warning: test file not found for client {client_id}, skipping.")
            continue

        
        main(
            load_8bit=True,
            base_model="/2014110093/zhangzy/FedDPA/data/Llama2-7B",
            lora_weights_path=lora_base_path,
            local_model_path=local_model_path,
            output_file=output_file,
            test_file=test_file,
            prompt_template="alpaca_short",
            batch_size=1,
            proj_scale=100.0,
            proj_threshold=0.005,
        )

if __name__ == "__main__":
    fire.Fire(run_all)