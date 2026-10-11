"""Учёт токенов и цены одного ответа агента (одного Runner.run_async).

Ответ агента — это несколько запросов к LLM (каждый шаг цикла). ADK кладёт usage_metadata
в итоговое (не partial) событие каждого запроса; суммируем их. Partial-события потока
пропускаем: в них либо нет usage, либо промежуточные числа, и сумма задвоилась бы.
"""

from __future__ import annotations

from dataclasses import dataclass

from google.adk.events import Event

from ai_course_agents.config import Settings


@dataclass
class Usage:
    llm_calls: int = 0
    input_tokens: int = 0       # все входные токены, включая взятые из кэша
    cached_tokens: int = 0      # из них — из кэша провайдера (дешевле)
    output_tokens: int = 0      # включая токены размышлений: DeepSeek считает их выходом
    thought_tokens: int = 0     # из них — на размышления (для интереса, если провайдер сообщает)

    def add(self, event: Event) -> None:
        u = event.usage_metadata
        if u is None or event.partial:
            return
        self.llm_calls += 1
        self.input_tokens += u.prompt_token_count or 0
        self.cached_tokens += u.cached_content_token_count or 0
        self.output_tokens += u.candidates_token_count or 0
        self.thought_tokens += u.thoughts_token_count or 0

    def cost_rub(self, settings: Settings) -> float:
        """Цена по тарифу из настроек (₽ за миллион токенов): свежий вход, вход из кэша
        и выход стоят по-разному."""
        fresh = self.input_tokens - self.cached_tokens
        return round(fresh / 1e6 * settings.price_in_rub
                     + self.cached_tokens / 1e6 * settings.price_cached_rub
                     + self.output_tokens / 1e6 * settings.price_out_rub, 4)

    def as_dict(self, settings: Settings) -> dict:
        return {"llm_calls": self.llm_calls, "input_tokens": self.input_tokens,
                "cached_tokens": self.cached_tokens,
                "output_tokens": self.output_tokens, "thought_tokens": self.thought_tokens,
                "cost_rub": self.cost_rub(settings)}
