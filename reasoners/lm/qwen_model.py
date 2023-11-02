import copy
import warnings
from typing import Optional, Union

import numpy as np
import torch
from transformers import StoppingCriteriaList

from reasoners import LanguageModel, GenerateOutput
from transformers import AutoTokenizer, GenerationConfig, PreTrainedTokenizer, AutoModelForCausalLM


class QwenModel(LanguageModel):

    def __init__(self, model_path: str, max_batch_size=1, max_new_tokens=None, max_length=2048, **kwargs):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, pad_token='<|endoftext|>')

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto"
        )
        self.device = self.model.device

        self.generation_config = GenerationConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        self.model.eval()

        self.max_batch_size = max_batch_size
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length

    def generate(self,
                 inputs: list[str],
                 max_length: Optional[int] = None,
                 max_new_tokens: Optional[int] = None,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 1.0,
                 num_return_sequences: int = 1,
                 eos_token_id: Union[None, str, int, list[str, int]] = None,
                 hide_input: bool = True,
                 output_log_probs: bool = False,
                 **kwargs) -> GenerateOutput:

        # unify eos_token
        if max_length is None:
            max_length = self.max_length
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        temperature = max(0.6, temperature)

        eos_token_id_input = copy.deepcopy(eos_token_id)
        eos_token_id = []

        if eos_token_id_input is not None:
            if not isinstance(eos_token_id_input, list):
                eos_token_id_input = [eos_token_id_input]
            for token in eos_token_id_input:
                if isinstance(token, str):
                    tokenized = self.tokenizer.encode(token, add_special_tokens=False)
                    if len(tokenized) != 1:
                        warnings.warn(f'the eos_token {repr(token)} is encoded into {tokenized} with length != 1, '
                                      f'using {tokenized[-1]} as the eos_token_id')
                    token = tokenized[-1]
                if isinstance(token, int):
                    eos_token_id.append(token)
                else:
                    warnings.warn(f'the eos_token {repr(token)} is neither str nor int, which is ignored')

        eos_token_id.append(self.generation_config.eos_token_id)

        generation_config = GenerationConfig(
            max_length=max_length,
            temperature=temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=eos_token_id,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
        )
        if max_new_tokens is not None:
            generation_config = GenerationConfig(
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=eos_token_id,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
            )

        if num_return_sequences > 1:
            assert len(inputs) == 1, 'num_return_sequences > 1 is not supported for multiple inputs'
            inputs = inputs * num_return_sequences
        decoded_list = []
        log_prob_list = []
        for start in range(0, len(inputs), self.max_batch_size):
            end = min(start + self.max_batch_size, len(inputs))
            encoded_inputs = self.tokenizer(inputs[start:end], return_tensors='pt', padding=True).to(self.model.device)
            with torch.inference_mode():
                generation_output = self.model.generate(
                    **encoded_inputs,
                    generation_config=generation_config,
                    output_scores=output_log_probs,
                    return_dict_in_generate=True,
                )
            decoded = self.tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
            if hide_input:
                for i in range(end - start):
                    decoded[i] = decoded[i][len(inputs[start + i]):]
            log_prob = None
            if output_log_probs:
                log_prob = generation_output.scores
                log_prob_list.extend(log_prob)
            decoded_list.extend(decoded)
        if not output_log_probs:
            log_prob_list = None

        return GenerateOutput(decoded_list, log_prob_list)

    @torch.no_grad()
    def get_next_token_logits(
            self,
            prompt: Union[str, list[str]],
            candidates: Union[list[str], list[list[str]]]) -> list[np.ndarray]:
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(candidates[0], str):
            candidates = [candidates] * len(prompt)
        cand_tokens = []
        for candidate in candidates:
            cand_tokens.append([])
            for cand in candidate:
                token = self.tokenizer.encode(cand, add_special_tokens=False)
                if len(token) != 1:
                    warnings.warn(f'candidate {cand} corresponds to {len(token)} instead of 1')
                cand_tokens[-1].append(token[1] if len(token) > 1 else token[0])

        bsz = len(prompt)
        assert bsz <= self.max_batch_size, (bsz, self.max_batch_size)

        tokens = self.tokenizer(prompt, return_tensors='pt', padding=True).to(self.device)
        with torch.no_grad():
            all_logits = self.model(**tokens, return_dict=True).logits[:, -1, :].squeeze(1)

        logits = []
        for case_logits, cand in zip(all_logits, cand_tokens):
            logits.append(case_logits[cand].cpu().numpy())
        return logits

    @torch.no_grad()
    def get_loglikelihood(self, prefix: str, contents: list[str], **kwargs) -> np.ndarray:

        bsz = len(contents)
        assert bsz <= self.max_batch_size, (bsz, self.max_batch_size)
        prompts_tokens = self.tokenizer(contents, return_tensors='pt', add_special_tokens=False, padding=True).to(
            self.device)
        prefix_tokens = self.tokenizer(prefix, return_tensors='pt', add_special_tokens=False, padding=True).input_ids[
            0].to(self.device)

        for prompt_tokens in prompts_tokens.input_ids:
            assert torch.all(prompt_tokens[: len(prefix_tokens)] == prefix_tokens), (prompt_tokens, prefix_tokens)

        tokens = prompts_tokens
        logits = self.model(**tokens, return_dict=True).logits
        tokens = prompts_tokens.input_ids
        acc_probs = torch.zeros(bsz).to(self.device)
        for i in range(len(prefix_tokens), tokens.shape[1]):
            probs = torch.softmax(logits[:, i - 1, :], dim=-1)
            for j in range(bsz):
                if tokens[j, i] != self.tokenizer.pad_token_id:
                    acc_probs[j] += torch.log(probs[j, tokens[j, i]])
        return acc_probs.cpu().numpy()
