from __future__ import annotations

from ._llm import LLMCallable, invoke_default_llm, parse_json_response, valid_confidence
from .schemas import QuestionSubtypePrediction, QuestionTypePrediction
from .subtype_catalog import QuestionSubtypeCatalog, load_subtype_catalog

_DEFAULT_CATALOG = load_subtype_catalog()
ALLOWED_SUBTYPES = _DEFAULT_CATALOG.allowed_codes()
SUBTYPE_NAMES = {item.code: item.name for item in _DEFAULT_CATALOG.items}


def _prediction_from_code(
    catalog: QuestionSubtypeCatalog, code: str, confidence: float
) -> QuestionSubtypePrediction:
    definition = catalog.by_code()[code]
    return QuestionSubtypePrediction(
        code=definition.code,
        officialCode=definition.officialCode,
        name=definition.name,
        confidence=confidence,
    )


def _fallback(
    catalog: QuestionSubtypeCatalog,
    question_type_code: str | None = None,
    confidence: float = 0.3,
) -> QuestionSubtypePrediction:
    if question_type_code in {"5", "6", "7", "8", "9"}:
        return _prediction_from_code(catalog, "-", confidence)
    allowed = catalog.allowed_codes().get(question_type_code or "", [])
    if allowed:
        return _prediction_from_code(catalog, allowed[0], confidence)
    return _prediction_from_code(catalog, "-", confidence)


class QuestionSubtypeClassifier:
    def __init__(
        self,
        llm: LLMCallable | None = None,
        catalog: QuestionSubtypeCatalog | None = None,
    ):
        self._llm = llm or invoke_default_llm
        self.catalog = catalog or _DEFAULT_CATALOG
        self._by_code = self.catalog.by_code()
        self.allowed_subtypes = self.catalog.allowed_codes()

    def classify(
        self, text: str, question_type: QuestionTypePrediction
    ) -> QuestionSubtypePrediction:
        return self.classify_top_k(text, question_type, k=1)[0]

    def classify_top_k(
        self, text: str, question_type: QuestionTypePrediction, k: int = 2
    ) -> list[QuestionSubtypePrediction]:
        if k < 1:
            raise ValueError("k must be positive")
        if question_type.code in {"5", "6", "7", "8", "9"}:
            return [_fallback(self.catalog, question_type.code, 1.0)]
        allowed = self.allowed_subtypes.get(question_type.code)
        if not allowed:
            return [_fallback(self.catalog, question_type.code)]
        allowed_lines = []
        for code in allowed:
            definition = self._by_code[code]
            allowed_lines.append(
                f"- code={definition.code}; officialCode={definition.officialCode}; "
                f"name={definition.name}"
            )
        prompt = f"""Определи официальный subtype для уже выбранного вида вопроса обращения.

Вид вопроса: {question_type.code} — {question_type.name}
Допустимые subtype:
{chr(10).join(allowed_lines)}

Логика выбора:
- Не выбирай subtype по ключевым словам механически; сопоставь смысл обращения с полным названием subtype.
- Для вида 1 отличай: совершенствование законов/НПА/работы органов от общего улучшения сферы жизни.
- Для вида 2 отличай просьбу о содействии в реализации прав/свобод от сообщения о нарушениях или недостатках в работе органов/должностных лиц.
- Для вида 3 отличай просьбу восстановить нарушенное право автора от жалобы на действие/бездействие должностных лиц, препятствие, незаконную обязанность или ответственность.
- Для вида 4 выбирай конкретную форму не-обращения: оценка, поздравление, приглашение, соболезнование, бессмысленный текст, материалы, просьба не по закону.
- Если несколько subtype близки, верни до {k} наиболее вероятных вариантов в порядке убывания уверенности.

Практические границы для вида 2:
- 2.1.1 выбирай, когда главный смысл — добиться помощи/содействия/решения конкретной жизненной ситуации автора или группы людей: жильё, выплата, ремонт, транспорт, доступность, медпомощь, коммунальная услуга.
- 2.2.1 выбирай, когда главный смысл — сообщить о нарушении закона как о факте и попросить проверить/пресечь нарушение.
- 2.2.2 выбирай, когда текст явно говорит о нарушении правил, порядка, требований, регламента, санитарных/ветеринарных/технических норм или иных НПА, но не обязательно прямо о нарушении закона.
- 2.2.3/2.2.4 выбирай, когда главный предмет — недостатки работы органа: не реагирует, плохо организует работу, не исполняет полномочия, дает формальные ответы. 2.2.3 — государственный орган; 2.2.4 — местное самоуправление.
- 2.2.5 выбирай, когда главный предмет — действия конкретного должностного лица, а не органа/системы в целом.
- Если в тексте есть и личная/жизненная проблема, и слова о нарушениях, не повышай автоматически до 2.2.x. Смотри, что является главным требованием: помочь решить ситуацию или зафиксировать нарушение.

Практические границы для вида 3:
- 3.1.1 выбирай, когда автор просит восстановить/защитить именно своё нарушенное право как конечный результат.
- 3.1.3 выбирай, когда речь шире о законном интересе автора, но не о конкретном субъективном праве.
- 3.2.1 выбирай, когда центр жалобы — действие/бездействие должностного или уполномоченного лица, из-за которого нарушены права/свободы.
- 3.2.2 выбирай, когда действия/бездействие создают препятствия реализации прав, но право ещё не обязательно окончательно нарушено.
- 3.2.3 выбирай только при незаконно возложенной обязанности.
- 3.2.4 выбирай только при незаконном привлечении к ответственности.

Few-shot ориентиры:
- Вид 2: "прошу помочь получить жилье/субсидию/услугу/реализовать право" -> обычно 2.1.1, если речь о правах автора.
- Вид 2: "прошу решить бытовую/транспортную/жилищную проблему, оказать содействие, принять меры по ситуации автора или жителей" -> чаще 2.1.1, даже если в тексте есть слова "жалоба", "нарушение" или "бездействие".
- Вид 2: "автобус не остановился, прошу разобраться и принять меры" -> 2.1.1. Это содействие в решении конкретной транспортной ситуации, а не сообщение о недостатках органа.
- Вид 2: "после ремонта дороги жители страдают от пыли, просят замеры, проверку и устранение проблемы" -> 2.1.1, если главный смысл — устранить проблему жителей; 2.2.1/2.2.2 возможны только если главный акцент на нарушении закона/норм.
- Вид 2: "прошу привести в порядок подъезд к больнице, убрать ямы, обеспечить доступность и благоустройство" -> 2.1.1. Упоминание учреждения или администрации не делает это 2.2.3.
- Вид 2: "сообщаю о нарушении закона, прошу проверить" -> 2.2.1; если речь об ином нормативном акте -> 2.2.2.
- Вид 2: "сообщаю о нарушении санитарных, ветеринарных, дорожных, технических правил/порядков и прошу проверить соблюдение требований" -> чаще 2.2.2, если речь именно о нормативных требованиях, а не о личной просьбе содействия.
- Вид 2: "плохо работает администрация/орган власти/учреждение, прошу принять меры" -> 2.2.3 для государственных органов или 2.2.4 для местного самоуправления.
- Вид 2: "конкретный сотрудник/должностное лицо плохо исполняет обязанности" -> 2.2.5.
- Не выбирай 2.2.3/2.2.4/2.2.5 только потому, что в тексте упомянут орган или должностное лицо. Для 2.2.x основной предмет должен быть именно недостаток работы органа/лица, а не сама жизненная проблема автора.
- Вид 2: "чиновники и ветеринарные службы допустили нарушения при изъятии скота, просим проверить и принять меры" -> 2.2.2/2.2.1 в зависимости от акцента на НПА или законах; не 2.1.1, если центр — нормативная законность действий.
- Вид 2: "орган власти формально отвечает, не организует решение проблемы, не исполняет полномочия" -> 2.2.3/2.2.4. Это недостатки работы органа, а не просто просьба о содействии.
- Вид 3: "прошу восстановить мое нарушенное право/защитить мое право" -> 3.1.1; если речь о свободе -> 3.1.2; если о законном интересе -> 3.1.3.
- Вид 3: "жалоба на действие или бездействие должностного лица, из-за которого нарушены права" -> 3.2.1.
- Вид 3: "зарплата присуждена/подтверждена, но решение не исполняется; прошу защитить право на выплату" -> 3.1.1, если центр — восстановление права автора на выплату.
- Вид 3: "должностное лицо отказало, затягивает или бездействует, из-за чего право автора не реализуется" -> 3.2.1 или 3.2.2; выбирай 3.1.1 только если просительная часть прямо про восстановление права как результат.
- Вид 3: "созданы препятствия реализации прав" -> 3.2.2; "незаконно возложена обязанность" -> 3.2.3; "незаконно привлечен к ответственности" -> 3.2.4.
- Вид 4: "спасибо/хорошо работает" -> 4.1; "поздравляю" -> 4.2; "приглашаю" -> 4.3; "соболезную" -> 4.4; бессвязный набор слов -> 4.5; материалы без просьбы -> 4.6; просьба, явно не основанная на законе -> 4.7.

Выбери до {k} допустимых внутренних code. Верни только JSON без markdown:
{{"candidates": [
  {{"code": "...", "confidence": 0.0, "reason": "кратко"}}
]}}

Текст обращения:
{text}"""
        try:
            payload = parse_json_response(self._llm(prompt))
            candidates = payload.get("candidates")
            if not isinstance(candidates, list):
                candidates = [payload]
            results: list[QuestionSubtypePrediction] = []
            seen: set[str] = set()
            for item in candidates:
                if not isinstance(item, dict):
                    continue
                code = str(item["code"])
                if code not in allowed or code in seen:
                    continue
                seen.add(code)
                definition = self._by_code[code]
                results.append(
                    QuestionSubtypePrediction(
                        code=code,
                        officialCode=definition.officialCode,
                        name=definition.name,
                        confidence=valid_confidence(item["confidence"]),
                    )
                )
                if len(results) >= k:
                    break
            if not results:
                raise ValueError("No valid subtype candidates")
            return results
        except Exception:
            return [_fallback(self.catalog, question_type.code)]
