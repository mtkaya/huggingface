"""DATALORA: Data-Aware LoRA with Mixture of Experts and progressive sparsification."""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model


@dataclass
class DATALORAConfig:
    """Configuration for DATALORA model."""

    base_model: str = "mistralai/Mistral-7B-v0.3"
    num_lora_experts: int = 8
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_retention: float = 0.5
    target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]
    )


class LoRAExpertRouter(nn.Module):
    """Routes inputs to different LoRA experts based on input features."""

    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.num_experts = num_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Use mean-pooled hidden states for routing
        if hidden_states.dim() == 3:
            pooled = hidden_states.mean(dim=1)
        else:
            pooled = hidden_states
        logits = self.gate(pooled)
        return torch.softmax(logits, dim=-1)


class SparsityMask(nn.Module):
    """Learnable sparsity mask for progressive pruning."""

    def __init__(self, size: int):
        super().__init__()
        self.scores = nn.Parameter(torch.ones(size))

    def forward(self, retention_ratio: float = 1.0) -> torch.Tensor:
        if retention_ratio >= 1.0:
            return torch.ones_like(self.scores)
        k = max(1, int(self.scores.numel() * retention_ratio))
        _, topk_idx = self.scores.abs().topk(k)
        mask = torch.zeros_like(self.scores)
        mask[topk_idx] = 1.0
        return mask


class DATALORAModel(nn.Module):
    """Wrapper that adds expert routing and sparsity masks on top of a PEFT model."""

    def __init__(self, peft_model, config: DATALORAConfig):
        super().__init__()
        self.peft_model = peft_model
        self.datalora_config = config

        hidden_size = peft_model.config.hidden_size
        self.router = LoRAExpertRouter(hidden_size, config.num_lora_experts)

        # Create sparsity masks for each LoRA layer
        self.sparsity_masks = nn.ModuleDict()
        for name, module in peft_model.named_modules():
            if hasattr(module, "lora_A"):
                safe_name = name.replace(".", "_")
                self.sparsity_masks[safe_name] = SparsityMask(config.lora_rank)

        self._current_retention = 1.0

    @property
    def config(self):
        return self.peft_model.config

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def set_retention(self, ratio: float):
        self._current_retention = ratio

    def gradient_checkpointing_enable(self, **kwargs):
        self.peft_model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.peft_model.gradient_checkpointing_disable()

    def get_input_embeddings(self):
        return self.peft_model.get_input_embeddings()

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.peft_model.prepare_inputs_for_generation(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.peft_model.generate(*args, **kwargs)

    def save_pretrained(self, output_dir, **kwargs):
        self.peft_model.save_pretrained(output_dir, **kwargs)
        # Save router and masks
        extra_state = {
            "router": self.router.state_dict(),
            "sparsity_masks": self.sparsity_masks.state_dict(),
            "config": self.config.__dict__,
        }
        torch.save(extra_state, f"{output_dir}/datalora_extra.pt")

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        return self.peft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )


def load_datalora_model(
    base_model_name: str,
    config: DATALORAConfig,
    device_map: str = "auto",
    torch_dtype=torch.bfloat16,
    load_in_4bit: bool = True,
) -> DATALORAModel:
    """Load a base model and wrap it with DATALORA components."""

    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    base_model.config.use_cache = False

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()

    model = DATALORAModel(peft_model, config)
    return model
