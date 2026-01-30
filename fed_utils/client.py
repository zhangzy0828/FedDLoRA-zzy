import transformers
from transformers import Trainer
import os
import torch.nn.functional as F
from datasets import load_dataset
import copy
from collections import OrderedDict
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from accelerate.hooks import remove_hook_from_module as _remove_hook_from_module
from peft import (
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
def safe_remove_hook_from_module(module, recurse=True):
    """
    Safe version of remove_hook_from_module that ignores missing _hf_hook.
    """
    try:
        _remove_hook_from_module(module, recurse=recurse)
    except AttributeError as e:
        if "_hf_hook" in str(e):
            # Ignore missing _hf_hook — module was not hooked
            pass
        else:
            raise
def neftune_post_forward_hook(module, input, output):
    """
    在 Embedding 层输出后添加均匀分布噪声
    output shape: (Batch, Seq_Len, Dim)
    """
    if module.training: 
        alpha = 5.0 
        dims = torch.tensor(output.size(1) * output.size(2))
        mag_norm = alpha / torch.sqrt(dims)
        
        epsilon = torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
        
        return output + epsilon
    return output

# # 蒸馏2
class KDTrainer(Trainer):
    def __init__(self, kd_alpha=0.1, kd_temperature=2.0, kd_log_steps=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kd_alpha = kd_alpha
        self.kd_temperature = kd_temperature
        self.kd_log_steps = kd_log_steps  # ✅ 新增：KD日志频率（step）
        self.kl_loss_fct = nn.KLDivLoss(reduction="none")

    def _safe_get_logits(self, outputs):
        if hasattr(outputs, "logits"):
            return outputs.logits
        elif isinstance(outputs, dict) and "logits" in outputs:
            return outputs["logits"]
        elif isinstance(outputs, (list, tuple)):
            for item in outputs:
                if isinstance(item, torch.Tensor) and item.dim() == 3:
                    return item
        raise ValueError("Could not extract logits from model outputs!")

    def _safe_get_loss(self, outputs, labels=None):
        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss
        elif isinstance(outputs, dict) and "loss" in outputs and outputs["loss"] is not None:
            return outputs["loss"]
        elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            if isinstance(outputs[0], torch.Tensor) and outputs[0].dim() == 0:
                return outputs[0]

        if labels is not None and self.label_smoother is not None:
            return self.label_smoother(outputs, labels)

        raise ValueError("Could not extract task loss from model outputs!")

    def _maybe_log_kd(self, task_loss, loss_kl, total_loss):

        if not self.model.training:
            return
        step = getattr(self.state, "global_step", 0)
        if self.kd_log_steps is None or self.kd_log_steps <= 0:
            return
        if step % self.kd_log_steps != 0:
            return
        t = float(task_loss.detach().cpu())
        k = float(loss_kl.detach().cpu()) if loss_kl is not None else 0.0
        tot = float(total_loss.detach().cpu())

        ratio = k / (t + 1e-12)
        self.log({
            "loss_task": t,
            "loss_kd": k,
            "loss_total": tot,
            "kd_ratio": ratio,  # KD 相对 task 的强度
            "kd_alpha": float(self.kd_alpha),
            "kd_T": float(self.kd_temperature),
        })

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        teacher_logits = None

        if self.kd_alpha > 0:
            was_training = model.training
            try:
                model.set_adapter("default")
                model.eval()
                with torch.no_grad():
                    teacher_inputs = {k: v for k, v in inputs.items() if k != "labels"}
                    teacher_outputs = model(**teacher_inputs)
                    teacher_logits = self._safe_get_logits(teacher_outputs)
            finally:
                model.set_adapter("local")
                if hasattr(model, "set_local"):
                    model.set_local(['default'], [1.0, 1.0])
                if was_training:
                    model.train()
        outputs = model(**inputs)
        student_logits = self._safe_get_logits(outputs)
        task_loss = self._safe_get_loss(outputs, labels)

        loss_kl = None
        if teacher_logits is not None:
            if student_logits.shape != teacher_logits.shape:
                if student_logits.size(-1) != teacher_logits.size(-1):
                    raise RuntimeError(
                        f"Vocab mismatch! Student: {student_logits.shape}, Teacher: {teacher_logits.shape}."
                    )
                L = min(student_logits.size(1), teacher_logits.size(1))
                student_logits = student_logits[:, :L, :]
                teacher_logits = teacher_logits[:, :L, :]
                if labels is not None:
                    labels = labels[:, :L]

            s_logits = student_logits.float()
            t_logits = teacher_logits.float()
            T = self.kd_temperature

            s_log_probs = F.log_softmax(s_logits / T, dim=-1)
            t_probs = F.softmax(t_logits / T, dim=-1)

            kl_raw = self.kl_loss_fct(s_log_probs, t_probs)      # [B,S,V]
            kl_per_token = kl_raw.sum(dim=-1)                    # [B,S]

            if labels is not None:
                mask = (labels != -100).float()
                masked_kl = kl_per_token * mask
                denom = mask.sum().clamp_min(1.0)
                loss_kl = masked_kl.sum() / denom
            else:
                loss_kl = kl_per_token.mean()

            loss_kl = loss_kl * (T ** 2)
            total_loss = task_loss + self.kd_alpha * loss_kl
        else:
            total_loss = task_loss
        self._maybe_log_kd(task_loss, loss_kl, total_loss)

        return (total_loss, outputs) if return_outputs else total_loss

class GeneralClient:
    def __init__(self, client_id, model, data_path, output_dir):
        self.client_id = client_id
        self.model = model
        self.neftune_handle = None 
        self.local_data_path = os.path.join(data_path, "local_training_{}.json".format(self.client_id))
        self.local_data = load_dataset("json", data_files=self.local_data_path)
        self.output_dir = output_dir
        self.local_output_dir = os.path.join(self.output_dir, "trainer_saved", "local_output_{}".format(self.client_id))

    def preprare_local_dataset(self, generate_and_tokenize_prompt, local_val_set_size):
        if local_val_set_size > 0:
            local_train_val = self.local_data["train"].train_test_split(
                test_size=local_val_set_size, shuffle=True, seed=42
            )
            self.local_train_dataset = (
                local_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            )
            self.local_eval_dataset = (
                local_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
            )
        else:
            self.local_train_dataset = self.local_data["train"].shuffle().map(generate_and_tokenize_prompt)
            self.local_eval_dataset = None
        self.local_val_set_size = local_val_set_size

    def build_local_trainer(self,
                            tokenizer,
                            local_micro_batch_size,
                            gradient_accumulation_steps,
                            local_num_epochs,
                            local_learning_rate,
                            group_by_length,
                            ddp,
                            kd_alpha=0.0, 
                            kd_temperature=2.0): 
        
        self.train_args = transformers.TrainingArguments(
            per_device_train_batch_size=local_micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=local_num_epochs,
            learning_rate=local_learning_rate,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            save_strategy="steps",
            save_steps=200,
            output_dir=self.local_output_dir,
            save_total_limit=1,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            dataloader_drop_last=False
        )

        self.local_trainer = KDTrainer(
            model=self.model,
            train_dataset=self.local_train_dataset,
            eval_dataset=self.local_eval_dataset,
            args=self.train_args,
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),            kd_alpha=kd_alpha,
            kd_temperature=kd_temperature
        )
   
    
    def set_local_dual_mode(self, epoch=0, prv=None):
        """
        设置 Dual LoRA 模式：
        1. 加载 Local Adapter 路径。
        2. 激活 Local Adapter。
        3. 挂载 'default' (Global) Adapter，并设置权重。
        """
        # 确定加载路径逻辑 (保持原样)
        if epoch == 1:
            single_output_dir = os.path.join(self.output_dir, str(epoch-1), "local_output_{}".format(self.client_id))
            tmp_path = None 
        else:
            if prv is None:
                single_output_dir = os.path.join(self.output_dir,str(epoch-1), "local_output_{}".format(self.client_id), 'local')
            else:
                tmp_epoch = 0
                if self.client_id in prv.keys():
                    tmp_epoch = prv[self.client_id]
                single_output_dir = os.path.join(self.output_dir,str(tmp_epoch), "local_output_{}".format(self.client_id), 'local')
            
            tmp_path = single_output_dir+'/pytorch_model.bin'
            if not os.path.exists(tmp_path):
                tmp_path = None

        self.model.add_local_model('local', tmp_path)
        self.model.set_adapter('local')

        self.model.set_local(['default'], [1.0, 1.0]) 
    def initiate_local_training(self, local=False, local_path=None, epoch=0, neftune_alpha=0.0):
        self.model.config.use_cache = False

        if local and neftune_alpha > 0:
            print(f"Enabling NEFTune noise for Local Training (alpha={neftune_alpha})...")

            def neftune_post_forward_hook(module, input, output):
                """
                在 Embedding 层输出后添加均匀分布噪声
                output shape: (Batch, Seq_Len, Dim)
                """
                if module.training: 
                    dims = torch.tensor(output.size(1) * output.size(2))
                    mag_norm = neftune_alpha / torch.sqrt(dims)
                
                    # 生成均匀分布噪声
                    if output.device != torch.device("cpu"):
                         epsilon = torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
                    else:
                         # 兼容 CPU 情况
                         epsilon = torch.zeros_like(output).uniform_(-mag_norm, mag_norm)

                    return output + epsilon
                return output

            # 2. 自动寻找 Embedding 层
            embed_layer = None
            if hasattr(self.model, "get_input_embeddings"):
                embed_layer = self.model.get_input_embeddings()
            elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "model") and hasattr(self.model.base_model.model, "embed_tokens"):
                embed_layer = self.model.base_model.model.embed_tokens
            elif hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
                embed_layer = self.model.model.embed_tokens
            
            # 3. 注册 Hook
            if embed_layer is not None:
                self.neftune_handle = embed_layer.register_forward_hook(neftune_post_forward_hook)
            else:
                print("Warning: Could not find Embedding layer, NEFTune skipped.")

        self.params_dict_old = copy.deepcopy(
            OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                        "default" in name))
        
        self.params_dict_new = OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                                           "default" in name)
        
        if local:
            # [CRITICAL CHANGE] 彻底解耦初始化逻辑
            
            if local_path is not None:
                single_output_dir = local_path
            else:
                target_epoch = max(0, epoch - 1) 
                single_output_dir = os.path.join(self.output_dir, str(target_epoch), "local_output_{}".format(self.client_id), 'local')
 
            
            local_weights_path = single_output_dir + "/pytorch_model.bin"

            if os.path.exists(local_weights_path):
                print(f"Loading existing LOCAL adapter from {local_weights_path}")
                local_adapters_weights = torch.load(local_weights_path)
                
                # 直接加载 Local 参数，不进行任何混合
                for k in self.params_dict_new.keys():
                    if "default" in k:
                        tmpk = k.split('default.')
                        k_local = tmpk[0] + tmpk[-1] # 构造 local key 名
                        if k_local in local_adapters_weights:
                            self.params_dict_new[k] = local_adapters_weights[k_local]
            else:
                print("No existing LOCAL adapter found. Initializing from GLOBAL (Cold Start).")

                pass

        new_adapter_weight = get_peft_model_state_dict(self.model, self.params_dict_new, "default")
        set_peft_model_state_dict(self.model, new_adapter_weight, "default")

   

    def train(self):
        self.local_trainer.train()

    def terminate_local_training(self, epoch, local_dataset_len_dict, previously_selected_clients_set, config, local=False):
        
        if self.neftune_handle is not None:
            self.neftune_handle.remove()
            self.neftune_handle = None
            if local:
                print("Disabled NEFTune noise (Local phase ended).")

        local_dataset_len_dict[self.client_id] = len(self.local_train_dataset)
        
        if local:
            single_output_dir = os.path.join(self.output_dir,str(epoch), "local_output_{}".format(self.client_id), 'local')
            adapter_name = 'local'
        else:
            single_output_dir = os.path.join(self.output_dir, str(epoch), "local_output_{}".format(self.client_id))
            adapter_name = 'default'
            
        os.makedirs(single_output_dir, exist_ok=True)
        
        
        new_adapter_weight = get_peft_model_state_dict(
                self.model, adapter_name=adapter_name
            )
        torch.save(new_adapter_weight, single_output_dir + "/pytorch_model.bin")
        config.save_pretrained(single_output_dir)

        older_adapter_weight = get_peft_model_state_dict(self.model, self.params_dict_old, "default")
        set_peft_model_state_dict(self.model, older_adapter_weight, "default")
        
        previously_selected_clients_set = previously_selected_clients_set | set({self.client_id})
        last_client_id = self.client_id

        safe_remove_hook_from_module(self.model, recurse=True)
        # =======================================================
        # 清理 Local Adapter
        if local:
            self.model.set_adapter("default")
            self.model.unset_local()
            self.model.delete_adapter('local')

        return self.model, local_dataset_len_dict, previously_selected_clients_set, last_client_id
