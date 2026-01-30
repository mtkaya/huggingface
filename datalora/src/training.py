"""DATALORA Trainer with 3-phase training: warmup, sparsification, hardening."""

from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import Trainer, TrainingArguments


@dataclass
class DATALORATrainingArguments(TrainingArguments):
    """Training arguments extended with DATALORA phase scheduling."""

    warmup_epochs: int = field(default=1, metadata={"help": "Epochs for warmup phase (full LoRA)"})
    sparsification_epochs: int = field(default=1, metadata={"help": "Epochs for progressive sparsification"})
    hardening_epochs: int = field(default=1, metadata={"help": "Epochs for hardening with fixed mask"})


class DATALORATrainer(Trainer):
    """Trainer that implements 3-phase DATALORA training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._phase = "warmup"

    def _get_phase(self, epoch: float) -> str:
        args = self.args
        if epoch < args.warmup_epochs:
            return "warmup"
        elif epoch < args.warmup_epochs + args.sparsification_epochs:
            return "sparsification"
        else:
            return "hardening"

    def _get_retention(self, epoch: float) -> float:
        args = self.args
        if epoch < args.warmup_epochs:
            return 1.0

        if epoch < args.warmup_epochs + args.sparsification_epochs:
            progress = (epoch - args.warmup_epochs) / max(args.sparsification_epochs, 1)
            target = self.model.datalora_config.target_retention if hasattr(self.model, "datalora_config") else 0.5
            return 1.0 - progress * (1.0 - target)

        target = self.model.datalora_config.target_retention if hasattr(self.model, "datalora_config") else 0.5
        return target

    def training_step(self, model, inputs, num_items_in_batch=None):
        if hasattr(self.state, "epoch") and self.state.epoch is not None:
            epoch = self.state.epoch
            phase = self._get_phase(epoch)
            retention = self._get_retention(epoch)

            if phase != self._phase:
                self._phase = phase
                print(f"\n>>> DATALORA phase: {phase} | retention: {retention:.2f}")

            if hasattr(model, "set_retention"):
                model.set_retention(retention)

        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir

        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(output_dir)
        else:
            super().save_model(output_dir, _internal_call=_internal_call)

        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
