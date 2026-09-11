from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .base import AudioLMWrapper, OptionScores, RoutingContext


@dataclass(frozen=True)
class QwenOmniConfig:
    model_id: str = "Qwen/Qwen2.5-Omni-3B"
    quantization: str = "nf4"
    thinker_only: bool = True
    device_map: str = "auto"
    attention_backend: str = "eager"
    torch_dtype: str = "float16"
    routing_layer: int = 0


class QwenOmniWrapper(AudioLMWrapper):
    """Qwen2.5-Omni Thinker wrapper with lazy GPU dependencies.

    `attention_backend='eager'` is required for exact QACR contribution routing.
    """

    def __init__(self, config: QwenOmniConfig):
        self.config = config
        self.model = None
        self.processor = None

    @staticmethod
    def build_conversation(audio_path: str, query: str):
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a careful audio reasoning assistant. Answer the user's query using only relevant evidence.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "path": str(audio_path)},
                    {"type": "text", "text": str(query)},
                ],
            },
        ]

    @staticmethod
    def find_subsequence_mask(input_ids: torch.Tensor, subsequence: Sequence[int]) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("MVP subsequence matching expects input_ids shape [1, seq]")
        ids = input_ids[0].tolist()
        sub = list(map(int, subsequence))
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if not sub:
            return mask
        starts = [i for i in range(len(ids) - len(sub) + 1) if ids[i : i + len(sub)] == sub]
        if not starts:
            raise ValueError("query token sequence was not found in processed input_ids")
        start = starts[-1]
        mask[0, start : start + len(sub)] = True
        return mask

    @staticmethod
    def extend_prefill_mask(mask: torch.Tensor, extra_tokens: int) -> torch.Tensor:
        if mask.ndim != 2:
            raise ValueError("mask must have shape [batch, sequence]")
        if extra_tokens < 0:
            raise ValueError("extra_tokens must be non-negative")
        if extra_tokens == 0:
            return mask
        pad = torch.zeros(mask.shape[0], extra_tokens, dtype=torch.bool, device=mask.device)
        return torch.cat([mask.to(dtype=torch.bool), pad], dim=1)

    @staticmethod
    def event_token_mask(
        audio_mask: torch.Tensor,
        *,
        duration_s: float,
        start_s: float,
        end_s: float,
    ) -> torch.Tensor:
        """Map a temporal source/event span onto audio placeholder tokens.

        This is a time-localization approximation, not source separation. It is only
        valid for events whose relevance can be represented by a temporal span.
        """
        if audio_mask.ndim != 2 or audio_mask.dtype is not torch.bool:
            raise ValueError("audio_mask must be boolean [batch, sequence]")
        if duration_s <= 0 or not (0 <= start_s < end_s <= duration_s + 1e-9):
            raise ValueError("event span must satisfy 0 <= start < end <= duration")
        out = torch.zeros_like(audio_mask)
        for b in range(audio_mask.shape[0]):
            pos = torch.where(audio_mask[b])[0]
            if pos.numel() == 0:
                continue
            centers = (torch.arange(pos.numel(), device=pos.device, dtype=torch.float32) + 0.5) / pos.numel() * float(duration_s)
            selected = (centers >= float(start_s)) & (centers < float(end_s))
            out[b, pos[selected]] = True
        return out

    def _dtype(self):
        if self.config.torch_dtype == "float16":
            return torch.float16
        if self.config.torch_dtype == "bfloat16":
            return torch.bfloat16
        if self.config.torch_dtype == "float32":
            return torch.float32
        raise ValueError(f"unsupported torch_dtype: {self.config.torch_dtype}")

    def load(self) -> None:
        try:
            from transformers import (
                BitsAndBytesConfig,
                Qwen2_5OmniProcessor,
                Qwen2_5OmniThinkerForConditionalGeneration,
            )
            if self.config.quantization == "nf4":
                import bitsandbytes  # noqa: F401
                import accelerate  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Install sar[gpu] to use QwenOmniWrapper (transformers/accelerate/bitsandbytes)."
            ) from exc

        quantization_config = None
        if self.config.quantization == "nf4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self._dtype(),
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.quantization != "none":
            raise ValueError(f"unsupported quantization: {self.config.quantization}")

        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            self.config.model_id,
            device_map=self.config.device_map,
            torch_dtype=self._dtype(),
            quantization_config=quantization_config,
            attn_implementation=self.config.attention_backend,
        )
        self.model.eval()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.config.model_id)

    @property
    def device(self):
        if self.model is None:
            raise RuntimeError("model is not loaded")
        return next(self.model.parameters()).device

    def prepare_inputs(self, audio_path: str, query: str):
        if self.model is None or self.processor is None:
            raise RuntimeError("call load() before preparing inputs")
        conversation = self.build_conversation(audio_path, query)
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        return inputs.to(self.device)

    @staticmethod
    def single_token_option_ids(tokenizer, options: Sequence[str]) -> list[int]:
        ids: list[int] = []
        for option in options:
            token_ids = tokenizer(option, add_special_tokens=False).input_ids
            if len(token_ids) != 1:
                raise ValueError(f"single token scoring requires every option to be one token; {option!r} -> {token_ids}")
            ids.append(int(token_ids[0]))
        return ids

    @staticmethod
    def append_candidate_inputs(base: dict, option_ids: torch.Tensor) -> dict:
        if option_ids.ndim != 2 or option_ids.shape[0] != 1:
            raise ValueError("option_ids must have shape [1, option_tokens]")
        out = dict(base)
        base_ids = base["input_ids"]
        if base_ids.ndim != 2 or base_ids.shape[0] != 1:
            raise ValueError("MVP candidate scoring expects batch size 1")
        out["input_ids"] = torch.cat([base_ids, option_ids.to(base_ids.device)], dim=1)
        if "attention_mask" in base:
            attention_mask = base["attention_mask"]
            extra = torch.ones(
                (attention_mask.shape[0], option_ids.shape[1]),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            out["attention_mask"] = torch.cat([attention_mask, extra], dim=1)
        out.pop("position_ids", None)
        return out

    @staticmethod
    def conditional_sequence_logprob(logits: torch.Tensor, target_ids: Sequence[int]) -> float:
        if logits.ndim != 2:
            raise ValueError("logits must have shape [sequence, vocab]")
        target = torch.as_tensor(target_ids, device=logits.device, dtype=torch.long)
        if target.numel() != logits.shape[0]:
            raise ValueError("target length must match logits sequence length")
        return float(torch.log_softmax(logits, dim=-1).gather(-1, target[:, None]).sum().item())

    def score_options(self, audio_path: str, query: str, options: Sequence[str]) -> OptionScores:
        if self.model is None or self.processor is None:
            raise RuntimeError("call load() before scoring")
        if not options:
            raise ValueError("options cannot be empty")
        base = self.prepare_inputs(audio_path, query)
        base_ids = base["input_ids"]
        base_len = base_ids.shape[1]
        scores: list[float] = []
        tokenizer = self.processor.tokenizer

        with torch.no_grad():
            for option in options:
                option_ids = tokenizer(option, add_special_tokens=False, return_tensors="pt").input_ids.to(base_ids.device)
                if option_ids.numel() == 0:
                    raise ValueError(f"option tokenized to empty sequence: {option!r}")
                forward_inputs = self.append_candidate_inputs(base, option_ids)
                outputs = self.model(**forward_inputs, use_cache=False)
                logits = outputs.logits[0]
                positions = torch.arange(base_len - 1, base_len - 1 + option_ids.shape[1], device=logits.device)
                token_logits = logits.index_select(0, positions)
                scores.append(self.conditional_sequence_logprob(token_logits, option_ids[0]))
        return OptionScores(options=list(options), logprobs=scores)

    def _base_masks(self, base_inputs: dict, query: str) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = base_inputs["input_ids"]
        audio_token_id = int(self.model.config.audio_token_index)
        audio_mask = input_ids.eq(audio_token_id)
        query_ids = self.processor.tokenizer(query, add_special_tokens=False).input_ids
        query_mask = self.find_subsequence_mask(input_ids, query_ids)
        return audio_mask, query_mask

    def score_single_token_options(
        self, audio_path: str, query: str, options: Sequence[str]
    ) -> OptionScores:
        if self.model is None or self.processor is None:
            raise RuntimeError("call load() before scoring")
        option_ids = self.single_token_option_ids(self.processor.tokenizer, options)
        inputs = self.prepare_inputs(audio_path, query)
        with torch.no_grad():
            outputs = self.model(**inputs, use_cache=False)
            logp = torch.log_softmax(outputs.logits[0, -1], dim=-1)
            scores = [float(logp[idx].item()) for idx in option_ids]
        return OptionScores(options=list(options), logprobs=scores)

    def score_single_token_options_qacr_tensors(
        self,
        audio_path: str,
        query: str,
        options: Sequence[str],
        router: torch.nn.Module,
        *,
        layer: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One-forward differentiable MCQ scoring for canonical single-token options."""
        if self.model is None or self.processor is None:
            raise RuntimeError("call load() before routed scoring")
        from .qwen_hook import QACRController, install_qacr_on_qwen_attention
        option_ids = self.single_token_option_ids(self.processor.tokenizer, options)
        inputs = self.prepare_inputs(audio_path, query)
        audio_mask, query_mask = self._base_masks(inputs, query)
        controller = QACRController(router=router)
        controller.set_masks(audio_mask.to(self.device), query_mask.to(self.device))
        routing_layer = int(self.config.routing_layer if layer is None else layer)
        restore = install_qacr_on_qwen_attention(self.get_text_attention_layer(routing_layer), controller)
        try:
            outputs = self.model(**inputs, use_cache=False)
            logits = outputs.logits[0, -1]
            ids = torch.tensor(option_ids, device=logits.device, dtype=torch.long)
            scores = torch.log_softmax(logits, dim=-1).index_select(0, ids)
            if controller.gate is None:
                raise RuntimeError("QACR hook did not compute a prefill gate")
            return scores, controller.gate, audio_mask.to(controller.gate.device)
        finally:
            restore()

    def score_options_qacr_tensors(
        self,
        audio_path: str,
        query: str,
        options: Sequence[str],
        router: torch.nn.Module,
        *,
        layer: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Differentiably score canonical options under QACR.

        Returns `(option_logprobs, gate, audio_mask)`. The backbone may be frozen;
        gradients still flow to the router through the patched attention contribution.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("call load() before routed scoring")
        if not options:
            raise ValueError("options cannot be empty")
        from .qwen_hook import QACRController, install_qacr_on_qwen_attention

        base = self.prepare_inputs(audio_path, query)
        base_ids = base["input_ids"]
        base_len = base_ids.shape[1]
        base_audio_mask, base_query_mask = self._base_masks(base, query)
        routing_layer = int(self.config.routing_layer if layer is None else layer)
        attn_module = self.get_text_attention_layer(routing_layer)
        controller = QACRController(router=router)
        restore = install_qacr_on_qwen_attention(attn_module, controller)
        tokenizer = self.processor.tokenizer
        scores: list[torch.Tensor] = []
        last_gate: torch.Tensor | None = None
        last_audio_mask: torch.Tensor | None = None
        try:
            for option in options:
                option_ids = tokenizer(option, add_special_tokens=False, return_tensors="pt").input_ids.to(base_ids.device)
                if option_ids.numel() == 0:
                    raise ValueError(f"option tokenized to empty sequence: {option!r}")
                forward_inputs = self.append_candidate_inputs(base, option_ids)
                audio_mask = self.extend_prefill_mask(base_audio_mask, option_ids.shape[1]).to(base_ids.device)
                query_mask = self.extend_prefill_mask(base_query_mask, option_ids.shape[1]).to(base_ids.device)
                controller.set_masks(audio_mask, query_mask)
                outputs = self.model(**forward_inputs, use_cache=False)
                logits = outputs.logits[0]
                positions = torch.arange(base_len - 1, base_len - 1 + option_ids.shape[1], device=logits.device)
                token_logits = logits.index_select(0, positions)
                logp = torch.log_softmax(token_logits, dim=-1).gather(-1, option_ids[0][:, None]).sum()
                scores.append(logp)
                if controller.gate is None:
                    raise RuntimeError("QACR hook did not compute a prefill gate")
                last_gate = controller.gate
                last_audio_mask = audio_mask
        finally:
            restore()
        assert last_gate is not None and last_audio_mask is not None
        return torch.stack(scores), last_gate, last_audio_mask

    def score_options_qacr(
        self,
        audio_path: str,
        query: str,
        options: Sequence[str],
        router: torch.nn.Module,
        *,
        layer: int | None = None,
    ) -> tuple[OptionScores, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            scores, gate, audio_mask = self.score_options_qacr_tensors(audio_path, query, options, router, layer=layer)
        return OptionScores(options=list(options), logprobs=[float(x) for x in scores.detach().cpu()]), gate.detach(), audio_mask.detach()

    def get_routing_context(self, audio_path: str, query: str) -> RoutingContext:
        if self.model is None or self.processor is None:
            raise RuntimeError("call load() before requesting routing context")
        inputs = self.prepare_inputs(audio_path, query)
        input_ids = inputs["input_ids"]
        audio_token_id = int(self.model.config.audio_token_index)
        audio_mask = input_ids.eq(audio_token_id)
        query_ids = self.processor.tokenizer(query, add_special_tokens=False).input_ids
        query_mask = self.find_subsequence_mask(input_ids, query_ids)
        with torch.no_grad():
            outputs = self.model(**inputs, use_cache=False, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        layer = int(self.config.routing_layer)
        if not 0 <= layer < len(hidden_states):
            raise IndexError(f"routing_layer {layer} outside available hidden states {len(hidden_states)}")
        h = hidden_states[layer]
        denom = query_mask.sum(dim=1, keepdim=True).clamp_min(1).to(h.dtype)
        query_repr = (h * query_mask.to(h.device)[..., None]).sum(dim=1) / denom
        return RoutingContext(
            query_repr=query_repr,
            audio_repr=h,
            audio_mask=audio_mask.to(h.device),
        )

    def get_text_attention_layer(self, layer: int):
        if self.model is None:
            raise RuntimeError("call load() first")
        return self.model.model.layers[layer].self_attn
