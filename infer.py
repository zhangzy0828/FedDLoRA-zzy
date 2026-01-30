import os
import fire
import torch
import transformers
import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from utils.prompter import Prompter
from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class EvalDataset(Dataset):
    def __init__(self, file, prompter, tokenizer):
        self.prompter = prompter
        self.tokenizer = tokenizer
        with open(file, 'r', encoding='utf-8') as f:
            # 兼容 JSONL 和 JSON List 格式
            try:
                self.data = [json.loads(line) for line in f.readlines() if line.strip()]
            except json.JSONDecodeError:
                f.seek(0)
                self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        instruction = sample['instruction']
        # 根据是否有 input 字段生成 prompt
        input_text = sample.get('input', None)
        prompt = self.prompter.generate_prompt(instruction, input_text)
        return prompt, sample
    
def writeFile(s, path):
    with open(path, 'a+', encoding='utf-8') as f1:
        f1.write(s + '\n')
def load_projection_matrix(local_model_path, device):
    try:
        print(f"Loading projection matrix directly from: {local_model_path}")
        state_dict = torch.load(local_model_path, map_location="cpu")
        
        target_key = None
        
        for key in state_dict.keys():
            if "lora_A" in key and ("layers.0." in key or "layers.00." in key):
                target_key = key
                break
        
        if target_key is None:
            min_layer = 999
            for key in state_dict.keys():
                if "lora_A" in key and "layers" in key:
                    parts = key.split('.')
                    try:
                        idx = int(parts[parts.index('layers') + 1])
                        if idx < min_layer:
                            min_layer = idx
                            target_key = key
                    except:
                        continue

        if target_key:
            print(f"Using projection layer: {target_key}") 
            weights = state_dict[target_key].float().to(device)
            return F.normalize(weights, dim=1)
        else:
            print("Warning: Could not find lora_A for Layer 0.")
            return None
            
    except Exception as e:
        print(f"Error loading projection matrix: {e}")
        return None

def get_dynamic_weight_by_projection(model, input_ids, local_A_normalized, device, scale_factor=20.0, threshold=0.005):

    if local_A_normalized is None:
        return 0.5 

    with torch.no_grad():
        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            embed_layer = model.base_model.model.model.embed_tokens
        elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            embed_layer = model.model.embed_tokens
        else:
            embed_layer = model.get_input_embeddings()
            
        embeds = embed_layer(input_ids) 
        input_vec = torch.mean(embeds, dim=1).squeeze()
        input_vec = input_vec.float()
        input_vec = F.normalize(input_vec, dim=0)

    # 计算投影
    projections = torch.matmul(local_A_normalized, input_vec)
    # 取平均绝对值
    raw_score = torch.mean(torch.abs(projections)).item()
    # print(f"DEBUG: Raw Projection Score = {raw_score:.6f}") 
    alpha = min(max((raw_score - threshold) * scale_factor, 0.0), 1.0)
    
    return alpha

def main(
    load_8bit: bool = True,
    base_model: str = "",
    lora_weights_path: str = "", 
    local_model_path: str = "",  
    output_file: str = "",
    test_file: str = "",
    prompt_template: str = "alpaca_short",
    batch_size: int = 1,
    proj_scale: float = 100.0,
    proj_threshold: float = 0.005,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert base_model, "Please specify a --base_model"
    
    if not os.path.exists(local_model_path):
        raise ValueError(f"Local model path not found: {local_model_path}")

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    print(f"Loading Base + Global from: {lora_weights_path}")
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(model, lora_weights_path, torch_dtype=torch.float16)
    else:
        model = LlamaForCausalLM.from_pretrained(base_model, device_map={"": device})
        model = PeftModel.from_pretrained(model, lora_weights_path)

    print(f"Loading Local Adapter from: {local_model_path}")
    model.add_local_model('local', local_model_path)

    if not load_8bit:
        model.half()
    
    model.eval()

    eval_dataset = EvalDataset(test_file, prompter, tokenizer)
    dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    print("Starting Inference...")

    for prompts, sample in tqdm(dataloader):
        prompt = prompts[0]
        # inputs = tokenizer(prompt, return_tensors="pt")
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        input_ids = inputs["input_ids"].to(device)

        alpha = 1.0  # <-- 强制设为 1.0
        
        model.set_adapter('local')
        model.set_local(['default'], [1.0, 1.0])

        generation_config = GenerationConfig(
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=128,
            )
        
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        response = prompter.get_response(output)
        if response.endswith('</s>'):
            response = response[:-4]

        res_dict = {
            'instruction': sample['instruction'][0],
            'input': sample.get('input', [''])[0],
            'output': sample.get('output', [''])[0],
            'answer': response,
            'category': sample.get('category', ['unknown'])[0],
            'dynamic_weight': alpha
        }
        writeFile(json.dumps(res_dict, ensure_ascii=False), output_file)

        model.set_adapter('default')

    print(f"Done. Results: {output_file}")

# if __name__ == "__main__":
#     fire.Fire(main)