"""
llm_client.py — Loads a local Hugging Face model and generates verdict predictions.

Mirrors embedding.py's mps/cpu/cuda device convention. Greedy decoding
(do_sample=False) for reproducibility — rerunning should not flip verdicts.
"""
from pathlib import Path

MAX_NEW_TOKENS = 1500  # bumped from 450: some models (observed on Qwen3.5-9B)
                        # ignore the "2-4 sentences" instruction entirely and
                        # produce long structured step-by-step reasoning before
                        # ever reaching the JSON verdict - 450 tokens cut them
                        # off mid-sentence every time (100% parse failure).
                        # Higher cap costs nothing for terser models (Qwen2.5-
                        # Coder, DeepSeek): max_new_tokens is just a ceiling,
                        # generation still stops at EOS once the model is done.


class LocalHFClient:
    def __init__(self, model_path: str | Path, device: str = "cpu"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = str(model_path)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.chat_template is None:
            raise ValueError(
                f"'{model_path}' has no chat template - it looks like a base "
                "(non-instruct) checkpoint. Use an instruct/chat model instead."
            )

        # NOTE: the dtype kwarg name below (torch_dtype) matches the transformers
        # API at the time this was written. If it's been renamed in the installed
        # version, the first attempt raises TypeError and this loop just falls
        # through to the "could not load" error below - check `transformers`
        # release notes for AutoModelForCausalLM.from_pretrained if that happens.
        last_error = None
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
                self.model.to(device)
                break
            except (RuntimeError, ValueError, TypeError) as e:
                last_error = e
                continue
        else:
            raise RuntimeError(
                f"Could not load '{model_path}' on device '{device}' with any dtype."
            ) from last_error

        self.model.eval()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
