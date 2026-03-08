# augmented/prompts.py
# 兼容层：实际 Prompt 内容统一由 core.prompt_registry 管理。

from src.core.prompt_registry import PROMPT_KEYS, core_prompt_registry


class PromptRegistry:
    def __init__(self) -> None:
        self._prompt_keys = {
            "default": PROMPT_KEYS.AUGMENTED_STANDARD,
            "standard": PROMPT_KEYS.AUGMENTED_STANDARD,
            "adversarial": PROMPT_KEYS.AUGMENTED_ADVERSARIAL,
            "mixed_pair": PROMPT_KEYS.AUGMENTED_MIXED_PAIR,
            "strict_negative_first": PROMPT_KEYS.AUGMENTED_STRICT_NEGATIVE_FIRST,
        }

    def get(self, profile: str) -> str:
        key = self._prompt_keys.get(profile, self._prompt_keys["default"])
        return core_prompt_registry.get(key)
