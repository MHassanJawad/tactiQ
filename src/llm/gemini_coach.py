import os
from typing import Dict, List


class GeminiCoach:
    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        api_key: str = None,
        temperature: float = 0.4,
        max_output_tokens: int = 300,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def _build_prompt(self, game_state: Dict, q_delta: float, retrieved_situations: List[Dict]) -> str:
        minute = game_state.get("minute", "unknown")
        score = game_state.get("score", "unknown")
        team = game_state.get("team", "unknown")
        event_type = game_state.get("event_type", "Unknown")

        retrieved_lines = []
        for i, row in enumerate(retrieved_situations[:5], start=1):
            retrieved_lines.append(
                f"{i}. minute={row.get('minute', 'NA')}, "
                f"event={row.get('event_type', 'Unknown')}, "
                f"next={row.get('next_event', 'Unknown')}, "
                f"similarity={row.get('similarity', 0.0):.3f}"
            )

        retrieved_block = "\n".join(retrieved_lines) if retrieved_lines else "No similar situations found."

        return (
            "You are a top-level football tactical analyst.\n"
            "Provide concise in-game coaching advice in plain language.\n"
            "Ground your recommendation in retrieved historical situations.\n"
            "Output format:\n"
            "1) Situation read\n"
            "2) Recommendation\n"
            "3) Risk if ignored\n\n"
            f"Game state:\n- minute: {minute}\n- score: {score}\n- team: {team}\n- current_event: {event_type}\n"
            f"- q_delta: {q_delta:.4f} (negative means current action is suboptimal)\n\n"
            f"Retrieved situations:\n{retrieved_block}\n"
        )

    def _fallback_advice(self, game_state: Dict, q_delta: float, retrieved_situations: List[Dict]) -> str:
        team = game_state.get("team", "Team")
        minute = game_state.get("minute", "unknown")
        if q_delta < -0.25:
            quality = "clear suboptimal decision signal"
        elif q_delta < -0.05:
            quality = "mildly suboptimal decision signal"
        else:
            quality = "neutral-to-good decision signal"

        top = retrieved_situations[0] if retrieved_situations else {}
        next_event_hint = top.get("next_event", "retain possession")
        sim = top.get("similarity", None)
        sim_text = f"{sim:.3f}" if isinstance(sim, (int, float)) else "NA"

        return (
            f"Situation read: Minute {minute}, {team} has a {quality} (q_delta={q_delta:.3f}).\n"
            f"Recommendation: In similar historical shapes (top similarity={sim_text}), the common follow-up is "
            f"'{next_event_hint}'. Prioritize a safer support option and improve spacing around the ball.\n"
            "Risk if ignored: forcing low-probability actions in this phase can increase turnover risk and lose "
            "territorial control in the next passage of play."
        )

    def advise(self, game_state: Dict, q_delta: float, retrieved_situations: List[Dict]) -> str:
        """
        Generates tactical advice using RAG retrieved context and q-delta signal.
        """
        if not self.api_key:
            return self._fallback_advice(game_state, q_delta, retrieved_situations)

        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model_name)
            prompt = self._build_prompt(game_state, q_delta, retrieved_situations)
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                },
            )
            text = getattr(response, "text", None)
            return text.strip() if text else self._fallback_advice(game_state, q_delta, retrieved_situations)
        except Exception:
            return self._fallback_advice(game_state, q_delta, retrieved_situations)

