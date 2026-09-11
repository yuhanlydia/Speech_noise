from __future__ import annotations

from typing import Sequence

import torch

from sar.data.blocks import (
    AcousticBlock,
    blocks_to_audio_token_mask,
    format_block_table,
    parse_audio_blocks,
)
from sar.methods.daa import format_focus_prompt, format_scan_prompt, parse_focus_declaration
from sar.models.base import OptionScores
from sar.models.daa_hook import (
    DAAController,
    install_daa_on_qwen_layers,
    post_audio_consumer_mask,
)
from sar.models.qwen_omni import QwenOmniWrapper


class QwenOmniDAAWrapper(QwenOmniWrapper):
    """Declarative Acoustic Attention extension for the Qwen2.5-Omni Thinker."""

    stop_on_complete_declaration = False

    @staticmethod
    def focus_tag(selected_ids: Sequence[str]) -> str:
        if not selected_ids:
            raise ValueError("DAA focus requires at least one block")
        return f'<focus_audio blocks="{",".join(selected_ids)}">'

    @classmethod
    def focused_query(
        cls,
        query: str,
        blocks: Sequence[AcousticBlock],
        selected_ids: Sequence[str],
    ) -> str:
        # Prompt-only and KV-mask conditions must receive exactly the same textual
        # information. Including the address table makes a declaration such as B2
        # meaningful even when the runtime mask is disabled.
        return (
            "Acoustic address table:\n"
            f"{format_block_table(blocks)}\n\n"
            f"Question: {query}\n"
            f"{cls.focus_tag(selected_ids)}\n"
            "The declaration above identifies the acoustic region selected for this query. "
            "Answer the question using the available acoustic evidence."
        )

    def generate_text(self, audio_path: str, prompt: str, *, max_new_tokens: int,
                      declaration_kind: str | None = None) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("call load() before generation")
        inputs = self.prepare_inputs(audio_path, prompt)
        prompt_len = int(inputs["input_ids"].shape[1])
        generation_kwargs = {}
        if self.stop_on_complete_declaration and declaration_kind is not None:
            from transformers import StoppingCriteriaList
            from sar.generation_stopping import DeclarationStoppingCriteria

            generation_kwargs["stopping_criteria"] = StoppingCriteriaList([
                DeclarationStoppingCriteria(self.processor.tokenizer, prompt_len, declaration_kind)
            ])
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                **generation_kwargs,
            )
        generated = output_ids[0, prompt_len:]
        return self.processor.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

    def declare_audio_blocks(
        self,
        audio_path: str,
        *,
        duration_s: float,
        max_blocks: int,
        max_new_tokens: int,
    ) -> tuple[list[AcousticBlock], str]:
        raw = self.generate_text(
            audio_path,
            format_scan_prompt(duration_s=duration_s, max_blocks=max_blocks),
            max_new_tokens=max_new_tokens,
            declaration_kind="blocks",
        )
        blocks = parse_audio_blocks(
            raw,
            duration_s=duration_s,
            max_blocks=max_blocks,
        )
        return blocks, raw

    def select_audio_blocks(
        self,
        audio_path: str,
        query: str,
        blocks: Sequence[AcousticBlock],
        *,
        max_selected: int,
        max_new_tokens: int,
    ) -> tuple[list[str], str]:
        raw = self.generate_text(
            audio_path,
            format_focus_prompt(query, blocks, max_selected=max_selected),
            max_new_tokens=max_new_tokens,
            declaration_kind="focus",
        )
        selected = parse_focus_declaration(
            raw,
            valid_block_ids={b.block_id for b in blocks},
            max_selected=max_selected,
        )
        return selected, raw

    def _resolve_daa_layers(self, layers: Sequence[int]) -> list:
        if self.model is None:
            raise RuntimeError("call load() before resolving DAA layers")
        all_layers = list(self.model.model.layers)
        if not layers:
            return [layer.self_attn for layer in all_layers]
        out = []
        for idx in layers:
            if idx < 0 or idx >= len(all_layers):
                raise IndexError(
                    f"DAA layer {idx} outside 0..{len(all_layers) - 1}"
                )
            out.append(all_layers[idx].self_attn)
        return out

    @staticmethod
    def _audio_duration(audio_path: str) -> float:
        try:
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "soundfile is required for DAA temporal block mapping"
            ) from exc
        return float(sf.info(audio_path).duration)

    def _daa_controller_for_inputs(
        self,
        inputs: dict,
        *,
        duration_s: float,
        blocks: Sequence[AcousticBlock],
        selected_ids: Sequence[str],
    ) -> DAAController:
        input_ids = inputs["input_ids"]
        audio_token_id = int(self.model.config.audio_token_index)
        audio_mask = input_ids.eq(audio_token_id)
        allowed = blocks_to_audio_token_mask(
            audio_mask,
            duration_s=duration_s,
            blocks=blocks,
            selected_ids=selected_ids,
        )
        if not torch.any(allowed):
            raise ValueError("selected DAA blocks mapped to zero audio tokens")
        consumer = post_audio_consumer_mask(audio_mask)
        controller = DAAController()
        controller.set_masks(audio_mask, allowed, consumer)
        return controller

    def score_single_token_options_daa(
        self,
        audio_path: str,
        query: str,
        options: Sequence[str],
        blocks: Sequence[AcousticBlock],
        selected_ids: Sequence[str],
        *,
        layers: Sequence[int],
        apply_kv_mask: bool = True,
    ) -> OptionScores:
        if self.model is None or self.processor is None:
            raise RuntimeError("call load() before DAA scoring")
        option_ids = self.single_token_option_ids(self.processor.tokenizer, options)
        focused_query = self.focused_query(query, blocks, selected_ids)
        inputs = self.prepare_inputs(audio_path, focused_query)

        restore = None
        if apply_kv_mask:
            controller = self._daa_controller_for_inputs(
                inputs,
                duration_s=self._audio_duration(audio_path),
                blocks=blocks,
                selected_ids=selected_ids,
            )
            restore = install_daa_on_qwen_layers(
                self._resolve_daa_layers(layers),
                controller,
            )
        try:
            with torch.no_grad():
                outputs = self.model(**inputs, use_cache=False)
                logp = torch.log_softmax(outputs.logits[0, -1], dim=-1)
                scores = [float(logp[idx].item()) for idx in option_ids]
        finally:
            if restore is not None:
                restore()
        return OptionScores(options=list(options), logprobs=scores)

    def generate_answer_daa(
        self,
        audio_path: str,
        query: str,
        blocks: Sequence[AcousticBlock],
        selected_ids: Sequence[str],
        *,
        layers: Sequence[int],
        max_new_tokens: int,
        apply_kv_mask: bool = True,
    ) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("call load() before DAA generation")
        focused_query = self.focused_query(query, blocks, selected_ids)
        inputs = self.prepare_inputs(audio_path, focused_query)
        prompt_len = int(inputs["input_ids"].shape[1])

        restore = None
        if apply_kv_mask:
            controller = self._daa_controller_for_inputs(
                inputs,
                duration_s=self._audio_duration(audio_path),
                blocks=blocks,
                selected_ids=selected_ids,
            )
            restore = install_daa_on_qwen_layers(
                self._resolve_daa_layers(layers),
                controller,
            )
        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
        finally:
            if restore is not None:
                restore()
        generated = output_ids[0, prompt_len:]
        return self.processor.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()
