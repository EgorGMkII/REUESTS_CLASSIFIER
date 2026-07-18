import asyncio
import hashlib
import json
import random
from typing import List, Dict, Any
import logging

logger = logging.getLogger("llm_service")


class LLMService:
    def __init__(self, client):
        self.client = client


    async def choose_question_type(self, text):

        prompt = f"""
        Ты — эксперт по классификации обращений граждан.

        Классифицируй текст в один из 9 классов:
        
        1. Предложение  
        2. Заявление  
        3. Жалоба  
        4. Не обращение  
        5. Запрос информации  
        6. Запрос  
        7. Запрос прокуратуры  
        8. Устное сообщение  
        9. Сообщение о коррупции  
        
        ---
        
        Верни JSON:
        
        {{
          "label": <число 1-9>,
          "label_name": "...",
          "confidence": 0-1,
          "reason": "коротко почему"
        }}
        
        ---
        
        Текст:
        \"\"\"
        {text}
        \"\"\"
        """

        response = await self.chat_completion(prompt)

        return self._safe_json_parse(response)

    async def chat_completion(
            self,
            prompt: str,
            system: str = "Return ONLY JSON.",
            model: str = "gpt-5-nano",
            temperature: float = 0.2,
    ):
        response = await self._with_retry(
            lambda: self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
            )
        )

        return response.choices[0].message.content

    async def _with_retry(self, fn, retries: int = 7):
        last_error = None

        for attempt in range(retries):
            try:
                return await fn()

            except Exception as e:
                last_error = e
                await asyncio.sleep((2 ** attempt) + random.random())

        raise RuntimeError(f"LLM failed after retries: {last_error}")

    def _safe_json_parse(self, text: str):
        if not text:
            logger.warning("Empty LLM response")
            return None

        try:
            return json.loads(text)

        except Exception:
            start = text.find("{")
            end = text.rfind("}")

            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except Exception:
                    pass

        logger.error("Invalid JSON from LLM:\n%s", text)
        return None
