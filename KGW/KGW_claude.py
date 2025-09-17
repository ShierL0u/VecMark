from typing import Any, Dict, Iterator, List, Mapping, Optional
import torch
from math import sqrt
from functools import partial
from transformers import LogitsProcessor, LogitsProcessorList
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import Field


class KGWConfig:
    """Config class for KGW algorithm (adapted for LangChain)"""
    
    def __init__(
        self,
        gamma: float = 0.5,
        delta: float = 2.0,
        hash_key: int = 15485863,
        z_threshold: float = 4.0,
        prefix_length: int = 1,
        f_scheme: str = "additive",
        window_scheme: str = "left",
        vocab_size: int = 50257,
        device: str = "cuda",
        gen_kwargs: Optional[Dict] = None
    ):
        self.gamma = gamma
        self.delta = delta
        self.hash_key = hash_key
        self.z_threshold = z_threshold
        self.prefix_length = prefix_length
        self.f_scheme = f_scheme
        self.window_scheme = window_scheme
        self.vocab_size = vocab_size
        self.device = device
        self.gen_kwargs = gen_kwargs or {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": None,  # 将在运行时设置
        }


class KGWUtils:
    """Utility class for KGW algorithm, contains helper functions."""

    def __init__(self, config: KGWConfig, *args, **kwargs) -> None:
        self.config = config
        self.rng = torch.Generator()
        self.rng.manual_seed(self.config.hash_key)
        # 创建时先在CPU上，后续会移动到正确设备
        self.prf = torch.randperm(self.config.vocab_size, generator=self.rng)
        self.f_scheme_map = {
            "time": self._f_time,
            "additive": self._f_additive,
            "skip": self._f_skip,
            "min": self._f_min
        }
        self.window_scheme_map = {
            "left": self._get_greenlist_ids_left,
            "self": self._get_greenlist_ids_self
        }

    def ensure_device(self, device):
        """确保所有张量都在正确的设备上"""
        if self.prf.device != torch.device(device):
            self.prf = self.prf.to(device)

    def _f(self, input_ids: torch.LongTensor) -> int:
        """Get the previous token."""
        return int(self.f_scheme_map[self.config.f_scheme](input_ids))
    
    def _f_time(self, input_ids: torch.LongTensor):
        """Get the previous token time."""
        time_result = 1
        for i in range(0, self.config.prefix_length):
            time_result *= input_ids[-1 - i].item()
        self.ensure_device(input_ids.device)
        return self.prf[time_result % self.config.vocab_size]
    
    def _f_additive(self, input_ids: torch.LongTensor):
        """Get the previous token additive."""
        additive_result = 0
        for i in range(0, self.config.prefix_length):
            additive_result += input_ids[-1 - i].item()
        self.ensure_device(input_ids.device)
        return self.prf[additive_result % self.config.vocab_size]
    
    def _f_skip(self, input_ids: torch.LongTensor):
        """Get the previous token skip."""
        self.ensure_device(input_ids.device)
        return self.prf[input_ids[- self.config.prefix_length].item()]

    def _f_min(self, input_ids: torch.LongTensor):
        """Get the previous token min."""
        self.ensure_device(input_ids.device)
        return min(self.prf[input_ids[-1 - i].item()] for i in range(0, self.config.prefix_length))
    
    def get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        """Get greenlist ids for the input_ids."""
        return self.window_scheme_map[self.config.window_scheme](input_ids)
    
    def _get_greenlist_ids_left(self, input_ids: torch.LongTensor) -> list[int]:
        """Get greenlist ids for the input_ids via leftHash scheme."""
        device = input_ids.device
        # 创建设备特定的生成器
        rng = torch.Generator(device=device)
        rng.manual_seed((self.config.hash_key * self._f(input_ids)) % self.config.vocab_size)
        greenlist_size = int(self.config.vocab_size * self.config.gamma)
        vocab_permutation = torch.randperm(self.config.vocab_size, device=device, generator=rng)
        greenlist_ids = vocab_permutation[:greenlist_size]
        return greenlist_ids
    
    def _get_greenlist_ids_self(self, input_ids: torch.LongTensor) -> list[int]:
        """Get greenlist ids for the input_ids via selfHash scheme."""
        device = input_ids.device
        greenlist_size = int(self.config.vocab_size * self.config.gamma)
        greenlist_ids = []
        f_x = self._f(input_ids)
        self.ensure_device(device)
        
        for k in range(0, self.config.vocab_size):
            h_k = f_x * int(self.prf[k])
            # 创建设备特定的生成器
            rng = torch.Generator(device=device)
            rng.manual_seed(h_k % self.config.vocab_size)
            vocab_permutation = torch.randperm(self.config.vocab_size, device=device, generator=rng)
            temp_greenlist_ids = vocab_permutation[:greenlist_size]
            if k in temp_greenlist_ids:
                greenlist_ids.append(k)
        return torch.tensor(greenlist_ids, device=device)
    
    def _compute_z_score(self, observed_count: int, T: int) -> float: 
        """Compute z-score for the given observed count and total tokens."""
        expected_count = self.config.gamma
        numer = observed_count - expected_count * T 
        denom = sqrt(T * expected_count * (1 - expected_count))  
        z = numer / denom
        return z
    
    def score_sequence(self, input_ids: torch.Tensor) -> tuple[float, list[int]]:
        """Score the input_ids and return z_score and green_token_flags."""
        num_tokens_scored = len(input_ids) - self.config.prefix_length
        if num_tokens_scored < 1:
            raise ValueError(
                f"Must have at least {1} token to score after "
                f"the first min_prefix_len={self.config.prefix_length} tokens required by the seeding scheme."
            )

        green_token_count = 0
        green_token_flags = [-1 for _ in range(self.config.prefix_length)]

        for idx in range(self.config.prefix_length, len(input_ids)):
            curr_token = input_ids[idx]
            greenlist_ids = self.get_greenlist_ids(input_ids[:idx])
            if curr_token in greenlist_ids:
                green_token_count += 1
                green_token_flags.append(1)
            else:
                green_token_flags.append(0)
        
        z_score = self._compute_z_score(green_token_count, num_tokens_scored)
        return z_score, green_token_flags


class KGWLogitsProcessor(LogitsProcessor):
    """LogitsProcessor for KGW algorithm, process logits to add watermark."""

    def __init__(self, config: KGWConfig, utils: KGWUtils, *args, **kwargs) -> None:
        self.config = config
        self.utils = utils

    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids: List[torch.LongTensor]) -> torch.BoolTensor:
        """Calculate greenlist mask for the given scores and greenlist token ids."""
        device = scores.device
        green_tokens_mask = torch.zeros_like(scores, device=device)
        
        for b_idx in range(len(greenlist_token_ids)):
            if greenlist_token_ids[b_idx] is not None:
                # 确保greenlist_token_ids在正确的设备上
                greenlist_ids = greenlist_token_ids[b_idx]
                if not isinstance(greenlist_ids, torch.Tensor):
                    greenlist_ids = torch.tensor(greenlist_ids, device=device)
                else:
                    greenlist_ids = greenlist_ids.to(device)
                
                green_tokens_mask[b_idx][greenlist_ids] = 1
        
        final_mask = green_tokens_mask.bool()
        return final_mask

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        """Bias the scores for the greenlist tokens."""
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to add watermark."""
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores

        device = scores.device
        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

        for b_idx in range(input_ids.shape[0]):
            # 确保input_ids在正确的设备上
            input_ids_batch = input_ids[b_idx].to(device)
            greenlist_ids = self.utils.get_greenlist_ids(input_ids_batch)
            batched_greenlist_ids[b_idx] = greenlist_ids

        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)

        scores = self._bias_greenlist_logits(
            scores=scores,
            greenlist_mask=green_tokens_mask,
            greenlist_bias=self.config.delta
        )
        return scores


class KGWWatermarkedLLM(LLM):
    """LangChain LLM implementation with KGW watermarking"""
    
    model: Any = Field(..., description="The underlying language model for text generation")
    tokenizer: Any = Field(..., description="Tokenizer for the language model")
    gamma: float = Field(0.5, description="Fraction of vocabulary in the greenlist")
    delta: float = Field(2.0, description="Bias applied to greenlist tokens")
    hash_key: int = Field(15485863, description="Seed for hash operations")
    z_threshold: float = Field(4.0, description="Z-score threshold for watermark detection")
    prefix_length: int = Field(1, description="Number of previous tokens considered for hashing")
    f_scheme: str = Field("additive", description="Hashing scheme for greenlist selection")
    window_scheme: str = Field("left", description="Window scheme for greenlist selection")
    device: str = Field("cuda", description="Device to run the model on")
    
    # Internal components (not exposed in constructor)
    config: KGWConfig = Field(default=None, exclude=True)
    utils: KGWUtils = Field(default=None, exclude=True)
    logits_processor: KGWLogitsProcessor = Field(default=None, exclude=True)
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        gamma: float = 0.5,
        delta: float = 2.0,
        hash_key: int = 15485863,
        z_threshold: float = 4.0,
        prefix_length: int = 1,
        f_scheme: str = "additive",
        window_scheme: str = "left",
        device: str = None,  # 改为None，自动检测
        **kwargs: Any
    ):
        # 自动检测模型设备
        if device is None:
            try:
                device = str(next(model.parameters()).device)
            except:
                device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize with Pydantic
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            gamma=gamma,
            delta=delta,
            hash_key=hash_key,
            z_threshold=z_threshold,
            prefix_length=prefix_length,
            f_scheme=f_scheme,
            window_scheme=window_scheme,
            device=device,
            **kwargs
        )
        
        # Initialize internal components
        vocab_size = len(tokenizer.get_vocab())
        
        # 设置生成参数，确保pad_token_id正确设置
        gen_kwargs = {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id,
        }
        gen_kwargs.update(kwargs.get("generation_kwargs", {}))
        
        self.config = KGWConfig(
            gamma=gamma,
            delta=delta,
            hash_key=hash_key,
            z_threshold=z_threshold,
            prefix_length=prefix_length,
            f_scheme=f_scheme,
            window_scheme=window_scheme,
            vocab_size=vocab_size,
            device=device,
            gen_kwargs=gen_kwargs
        )
        self.utils = KGWUtils(self.config)
        self.logits_processor = KGWLogitsProcessor(self.config, self.utils)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Generate watermarked text from prompt."""
        if stop is not None:
            raise ValueError("stop kwargs are not permitted for this implementation.")
        
        device = self.config.device
        
        # 确保模型在正确的设备上（虽然通常已经在了）
        # self.model.to(device)  # 注释掉，因为模型已经分布在多个GPU上
        
        # 合并生成参数
        generation_config = {**self.config.gen_kwargs, **kwargs}
        
        generate_with_watermark = partial(
            self.model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            **generation_config
        )
        
        # 编码输入并移动到正确设备
        encoded_prompt = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        # 移动到模型的第一个参数所在的设备
        model_device = next(self.model.parameters()).device
        encoded_prompt = {k: v.to(model_device) for k, v in encoded_prompt.items()}
        
        # 生成输出
        encoded_output = generate_with_watermark(**encoded_prompt)
        
        # 只解码新生成的部分
        input_length = encoded_prompt['input_ids'].shape[-1]
        generated_tokens = encoded_output[0][input_length:]
        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return output_text

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream watermarked text generation token by token."""
        if stop is not None:
            raise ValueError("stop kwargs are not permitted for this implementation.")
        
        device = self.config.device
        model_device = next(self.model.parameters()).device
        
        generate_with_watermark = partial(
            self.model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            **{**self.config.gen_kwargs, **{"max_new_tokens": 1}}
        )
        
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(model_device)
        generated_ids = input_ids.clone()
        
        while True:
            output = generate_with_watermark(input_ids=generated_ids)
            new_token = output[0, -1]
            generated_ids = torch.cat([generated_ids, new_token.unsqueeze(0)], dim=1)
            
            chunk = GenerationChunk(text=self.tokenizer.decode(new_token, skip_special_tokens=True))
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                
            yield chunk
            
            if new_token == self.tokenizer.eos_token_id:
                break

    def detect_watermark(self, text: str, return_dict: bool = True) -> Any:
        """Detect watermark in the given text."""
        model_device = next(self.model.parameters()).device
        encoded_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(model_device)
        
        # z_score, _ = self.utils.score_sequence(encoded_text)
        # is_watermarked = z_score > self.config.z_threshold
         # 添加长度检查
        if len(encoded_text) <= self.config.prefix_length:
            if return_dict:
                return {"is_watermarked": False, "score": 0.0, "error": "text_too_short"}
            else:
                return (False, 0.0)
        
        try:
            z_score, _ = self.utils.score_sequence(encoded_text)
            is_watermarked = z_score > self.config.z_threshold
        except ValueError as e:
            if "Must have at least" in str(e):
                if return_dict:
                    return {"is_watermarked": False, "score": 0.0, "error": str(e)}
                else:
                    return (False, 0.0)
            raise

        if return_dict:
            return {"is_watermarked": is_watermarked, "score": z_score}
        else:
            return (is_watermarked, z_score)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters."""
        return {
            "model_type": "KGWWatermarkedLLM",
            "gamma": self.config.gamma,
            "delta": self.config.delta,
            "hash_key": self.config.hash_key,
            "f_scheme": self.config.f_scheme,
            "window_scheme": self.config.window_scheme
        }

    @property
    def _llm_type(self) -> str:
        """Return LLM type."""
        return "kgw_watermarked_llm"