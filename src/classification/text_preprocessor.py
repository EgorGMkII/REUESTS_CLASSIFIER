from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ._llm import LLMCallable, invoke_preprocessor_llm, parse_json_response


DEFAULT_SECTION_HINTS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "classifiers" / "section_retrieval_hints.json"
)
DEFAULT_TOPIC_ANCHORS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "classifiers" / "retrieval_topic_anchors.json"
)
VALID_SECTIONS = {"0001", "0002", "0003", "0004", "0005"}
MAX_RETRIEVAL_QUERIES = 4
MAX_QUERY_CHARS = 500


@dataclass(frozen=True)
class RetrievalQuery:
    section: str | None
    query: str


@dataclass(frozen=True)
class PreparedText:
    decision_text: str
    retrieval_queries: list[RetrievalQuery]
    type_decision_text: str = ""


class TextPreprocessor(Protocol):
    def prepare(self, text: str) -> str: ...


def compact_ocr_text(text: str, max_chars: int = 12_000) -> str:
    value = text or ""
    value = value.replace("\ufeff", " ").replace("\xa0", " ")
    value = re.sub(r"={10,}", " ", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    value = value.strip()
    if len(value) > max_chars:
        return value[:max_chars].rsplit(" ", 1)[0].strip()
    return value


def _load_section_hints(path: Path = DEFAULT_SECTION_HINTS_PATH) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _load_topic_anchors(path: Path = DEFAULT_TOPIC_ANCHORS_PATH) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _clean_text_list(value: object, max_items: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        cleaned = compact_ocr_text(item, 240)
        if cleaned:
            result.append(cleaned)
        if len(result) >= max_items:
            break
    return result


def _clean_decomposition(payload: dict[str, object]) -> dict[str, object]:
    anchor_id = payload.get("anchorId")
    main_subject = payload.get("mainSubject")
    domain = payload.get("domain")
    primary_problem = payload.get("primaryProblem")
    return {
        "anchorId": compact_ocr_text(anchor_id, 120) if isinstance(anchor_id, str) else "",
        "mainSubject": compact_ocr_text(main_subject, 240) if isinstance(main_subject, str) else "",
        "domain": compact_ocr_text(domain, 240) if isinstance(domain, str) else "",
        "primaryProblem": compact_ocr_text(primary_problem, 360) if isinstance(primary_problem, str) else "",
        "secondaryProblems": _clean_text_list(payload.get("secondaryProblems")),
        "detailsToDrop": _clean_text_list(payload.get("detailsToDrop")),
    }


def _fallback_query(text: str) -> list[RetrievalQuery]:
    query = compact_ocr_text(text, MAX_QUERY_CHARS)
    return [RetrievalQuery(section=None, query=query)] if query else []


def _clean_retrieval_queries(payload: object, fallback_text: str) -> list[RetrievalQuery]:
    if not isinstance(payload, list):
        return _fallback_query(fallback_text)
    queries: list[RetrievalQuery] = []
    seen: set[tuple[str | None, str]] = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        query = item.get("query")
        if not isinstance(query, str):
            continue
        query = compact_ocr_text(query, MAX_QUERY_CHARS)
        if len(query) < 8:
            continue
        section = item.get("section")
        if section is not None:
            section = str(section)
            if section not in VALID_SECTIONS:
                continue
        key = (section, query.lower())
        if key in seen:
            continue
        seen.add(key)
        queries.append(RetrievalQuery(section=section, query=query))
        if len(queries) >= MAX_RETRIEVAL_QUERIES:
            break
    return queries or _fallback_query(fallback_text)


class BasicTextPreprocessor:
    def __init__(self):
        self.last_retrieval_queries: list[RetrievalQuery] = []
        self.last_type_decision_text = ""

    def prepare(self, text: str) -> str:
        value = compact_ocr_text(text)
        self.last_retrieval_queries = _fallback_query(value)
        self.last_type_decision_text = value
        return value


class LLMTextPreprocessor:
    def __init__(
        self,
        llm: LLMCallable | None = None,
        fallback: TextPreprocessor | None = None,
        max_input_chars: int = 12_000,
        max_output_chars: int = 6_000,
        section_hints_path: Path = DEFAULT_SECTION_HINTS_PATH,
        topic_anchors_path: Path = DEFAULT_TOPIC_ANCHORS_PATH,
    ):
        self._llm = llm or invoke_preprocessor_llm
        self._fallback = fallback or BasicTextPreprocessor()
        self._max_input_chars = max_input_chars
        self._max_output_chars = max_output_chars
        self._section_hints = _load_section_hints(section_hints_path)
        self._topic_anchors = _load_topic_anchors(topic_anchors_path)
        self.last_retrieval_queries: list[RetrievalQuery] = []
        self.last_retrieval_decomposition: dict[str, object] = {}
        self.last_type_decision_text = ""

    def _prompt(self, fallback_text: str) -> str:
        hints = json.dumps(self._section_hints, ensure_ascii=False, indent=2)
        anchors = json.dumps(self._topic_anchors, ensure_ascii=False, indent=2)
        return f"""Ты готовишь OCR-текст обращения гражданина для классификации и поиска тем в официальном справочнике.

Нужно вернуть две разные вещи:
1. decisionText — краткая смысловая выжимка для выбора вида вопроса и subtype.
2. typeDecisionText — выжимка для выбора официального вида вопроса и subtype.
3. retrievalQueries — короткие поисковые запросы для тематического retriever.

Справочник верхних секций и подсказки:
{hints}

Компактный список тематических якорей для retrieval:
{anchors}

Правила для decisionText:
- сожми обращение до 6–12 содержательных предложений;
- сохрани главную проблему, объекты обращения, органы/организации, места, действия/бездействие, просьбу автора и ключевые тематические слова;
- явно сохрани административный смысл: автор просит сведения, содействие, проверку, принятие мер, восстановление права, отмену решения, ремонт, выплату, услугу и т.п.;
- если исходный OCR/текст содержит явный маркер "Запрос прокуратуры" в начале
  или в шапке документа, обязательно оставь фразу "Запрос прокуратуры." в
  начале decisionText без перефразирования;
- убери приветствия, подписи, персональные данные, повторы, технический мусор OCR, политические вступления и длинные цитаты законов;
- не добавляй коды тем, не выбирай итоговый вид вопроса, не придумывай новые факты.

Правила для typeDecisionText:
- сделай 8–15 предложений, можно подробнее decisionText;
- сохрани административный контекст, который влияет на вид обращения: кто автор, форма документа, от чьего имени подано, кому адресовано, основная просьба/требование;
- если исходный OCR/текст начинается с "Запрос прокуратуры", первая фраза
  typeDecisionText тоже должна начинаться строго с "Запрос прокуратуры.".
  Не заменяй это на "обращение поступило в прокуратуру";
- если видишь "Запрос прокуратуры" в исходном тексте, всегда сохрани эту точную
  фразу в typeDecisionText. Не переформулируй как "обращение в прокуратуру",
  "обращение поступило в прокуратуру", "запрос о проверке" или "прокурорская
  проверка";
- маркер "Запрос прокуратуры" является служебным признаком формы документа, его
  нельзя удалять, обобщать или заменять даже если дальше текст пересказывает
  обращение гражданина;
- явно сохрани признаки: "прокуратура запрашивает/просит", "служебный/межведомственный запрос", "заявитель просит сведения", "заявитель просит содействие/меры", "заявитель обжалует решение/действие/бездействие";
- если автор просит "сообщить/разъяснить", обязательно поясни: это самостоятельный запрос сведений или сведения нужны для решения конкретной проблемы;
- если есть жалобные слова, обязательно поясни: автор обжалует конкретное решение/действие или только сообщает о проблеме и просит меры;
- не выкидывай шапку/форму документа, если она помогает отличить запрос прокуратуры, служебный запрос, обычное обращение гражданина, заявление, жалобу или запрос информации;
- не выбирай итоговый код вида вопроса, только сохрани признаки для следующего классификатора;
- не называй итоговый вид обращения словами "это жалоба", "это заявление", "это запрос информации", "по форме это жалоба", "форма документа — заявление";
- вместо ярлыков описывай наблюдаемые признаки: автор просит проверить, автор просит принять меры, автор просит содействие, автор просит сведения, автор оспаривает конкретное решение/действие/бездействие, сведения нужны для жилищной/имущественной/социальной проблемы;
- если текст содержит слово "жалоба", можно написать "в тексте есть жалобные формулировки", но нельзя делать вывод "это жалоба";
- не злоупотребляй словами "обжалует" и "оспаривает": используй их только если автор явно просит отменить конкретный акт/решение/отказ, признать действие незаконным или восстановить нарушенное право;
- если автор в основном просит проверку, меры реагирования, разобраться, устранить проблему, выплату, ремонт или содействие, описывай это нейтрально как "автор просит проверить/принять меры/оказать содействие", даже если текст эмоциональный;
- если текст содержит "прошу сообщить", можно написать "автор просит сообщить сведения", но нельзя делать вывод "это запрос информации".
- если обращение коллективное или публичное и авторы просят проверить законность,
  остановить практику, дать оценку, принять меры реагирования или направить
  протест/представление, описывай это как просьбу о проверке и реагировании,
  а не как индивидуальное обжалование;
- слова "отменить", "опротестовать", "защитить права" не превращают текст в
  обжалование автоматически: отдельно укажи, есть ли конкретный отказ/решение
  именно в отношении заявителя и просьба восстановить его конкретное право;
- если речь о зарплате, судебных решениях, исполнительных документах,
  длительном неисполнении и фактическом невосстановлении права на выплату,
  явно сохрани это как признак требования защитить/восстановить уже нарушенное
  трудовое право.

Правила для retrievalQueries:
- сначала сделай внутреннюю декомпозицию: anchorId, mainSubject, domain, primaryProblem, secondaryProblems, detailsToDrop;
- anchorId выбирай из списка тематических якорей; если нет подходящего — оставь пустым, но mainSubject всё равно сформулируй;
- mainSubject — главный предмет справочника, например: эксплуатация автомобильных дорог, выплата заработной платы, улучшение жилищных условий, животноводство, лечение и медицинская помощь;
- primaryProblem — главная проблема внутри предмета, без адресов и реквизитов;
- secondaryProblems — только сопутствующие проблемы, они не должны вытеснять главный предмет;
- detailsToDrop — частные детали, которые нельзя превращать в retrieval query;
- выбери 1 section; если обращение реально смешанное или есть сильная неопределенность — выбери 2 sections;
- если выбрана 1 section — верни 3 retrievalQueries;
- если выбраны 2 sections — верни по 2 retrievalQueries на каждую section;
- не возвращай больше 4 retrievalQueries;
- retrieval query — это не пересказ обращения, а запрос к справочнику тем;
- пиши так, будто ищешь название официальной тематики, а не конкретный документ;
- каждый query должен быть коротким обобщением проблемы языком классификатора;
- не указывай leaf-коды тем;
- не используй section как жесткое решение: section только помогает подобрать лексику;
- разные query должны искать под разными углами: официальная лексика, бытовая формулировка, объекты/действия/участники.
- не тащи в retrievalQueries ФИО, адреса, названия улиц, населенные пункты, номера домов, даты, номера документов, номера дел, суммы, марки материалов, номера дорожных знаков и прочие реквизиты;
- исключение: оставляй только тематически важный объект общего типа — дорога, школа, больница, автобус, полигон ТБО, СНТ, жилье, скот;
- технические детали обобщай: "ЩПС С-1" -> "дорожное покрытие"; "знак 5.21" -> "дорожные знаки"; "ул. Мельничная" -> "автомобильная дорога";
- избегай слишком общих фраз без предмета: "нарушение прав", "просьба о содействии", "незаконные действия";
- полезный query = отрасль + объект + проблема/действие + официальная лексика темы.
- каждый query должен начинаться с mainSubject или близкой фразы из preferredPhrases выбранного anchor;
- в первом query всегда отражай главный предмет обращения, а не вторичный эффект, адрес, меру реагирования или правовую процедуру;
- если есть причина и последствия, первый query пиши про причину/предмет, второй можно про последствия;
- если есть объект и требуемая мера, первый query пиши про объект/состояние объекта, а не про меру;
- если есть отрасль и жалоба на должностных лиц, первый query пиши про отраслевой предмет, а не про прокуратуру/проверку/превышение полномочий;
- если есть бытовые детали, обобщай их до класса тем: "пыль из-за покрытия дороги" -> "эксплуатация и содержание автомобильных дорог"; "нет выплаты подрядчиком" -> "невыплата заработной платы"; "забой коров" -> "животноводство, ветеринарные меры"; "дачный участок в СНТ" -> "садоводческие товарищества, дачные участки".

Few-shot:
- Текст: "Автобус маршрута 68 не остановился на остановке, пришлось далеко идти."
  Плохо: "маршрут 68 остановка Карьер Борок 7 июня автобус К 573 КЕ"
  Хорошо: section 0003, query: "транспортное обслуживание населения пассажирские перевозки автобус не остановился на остановке"
- Текст: "Массовая невыплата заработной платы на строительстве завода подрядчиком."
  Хорошо: section 0002, query: "трудовые права невыплата заработной платы задержка зарплаты работники подрядчик"
- Текст: "Незаконный забой коров, изъятие скота у фермеров, ветеринарные меры."
  Хорошо: section 0003, query: "сельское хозяйство животноводство крупный рогатый скот забой скота ветеринарные меры"
- Текст: "Инвалид спрашивает, положена ли субсидия на съем или приобретение жилья, дом без удобств."
  Хорошо: section 0005, query: "жилищные условия субсидия на жилье предоставление жилья дом без удобств социальная поддержка жильем"
- Текст: "Полиция, прокуратура, превышение полномочий, регистрация по месту жительства."
  Хорошо: section 0004, query: "полиция прокуратура превышение полномочий регистрация по месту жительства законность"
- Текст: "После ремонта дороги щебеночно-песчаной смесью на улице Мельничная в Чулыме дома покрывает пылью, жители просят проверку и меры."
  Плохо: "улица Мельничная Чулым ЩПС С-1 знаки 5.21 5.22 запрет грузового транспорта жилая зона"
  Хорошо: section 0003, query: "эксплуатация и сохранность автомобильных дорог содержание автомобильной дороги ремонт дорожного покрытия"
  Дополнительно: section 0003, query: "пыль от дороги загрязнение воздуха из-за дорожного покрытия"
- Текст: "Жители СНТ жалуются на повреждение дачного строения, отказ правления помочь, взносы и имущество участка."
  Плохо: "река Иня фамилия председателя наличные деньги конкретный участок"
  Хорошо: section 0005, query: "садоводческое товарищество дачный участок имущество СНТ содержание территории управление товариществом"
- Текст: "Из-за полигона ТБО рядом с жилыми домами запах, фильтрат, пожары, загрязнение почвы и воды."
  Плохо: "детский сад 300 метров ручей Норниста решения суда дата проверки"
  Хорошо: section 0003, query: "полигоны бытовых отходов переработка отходов эксплуатация полигона ТБО"
  Дополнительно: section 0003, query: "загрязнение окружающей среды отходы фильтрат пожары полигон ТБО"
- Текст: "Работникам подрядчика на строительстве завода массово не выплачивают заработную плату."
  Плохо: "строительство завода литий-ионные батареи сроки ввода объекта прокуратура уголовное дело"
  Хорошо: section 0002, query: "трудовые права невыплата заработной платы задержка зарплаты работники подрядчик"
- Текст: "При борьбе с заболеванием животных у фермеров изымают и забивают коров, владельцы оспаривают ветеринарные меры."
  Плохо: "превышение полномочий прокуратура следственный комитет частная территория"
  Хорошо: section 0003, query: "сельское хозяйство животноводство крупный рогатый скот забой скота ветеринарные меры"
- Текст: "Житель просит жилье по соцнайму или субсидию, живет в доме без удобств, семья нуждается в улучшении жилищных условий."
  Плохо: "диагнозы адрес станция ФИО родственников"
  Хорошо: section 0005, query: "улучшение жилищных условий предоставление жилья социальный найм нуждающиеся в жилых помещениях"
  Дополнительно: section 0005, query: "субсидия на жилье предоставление жилого помещения жилищная поддержка"

Верни только JSON без markdown:
{{
  "decisionText": "краткая смысловая выжимка обращения",
  "typeDecisionText": "выжимка с административным контекстом для выбора вида вопроса",
  "anchorId": "id выбранного тематического якоря или пустая строка",
  "mainSubject": "главный предмет справочника",
  "domain": "краткая сфера обращения",
  "primaryProblem": "главная проблема без реквизитов",
  "secondaryProblems": ["сопутствующая проблема"],
  "detailsToDrop": ["частная деталь, которую нельзя тащить в query"],
  "retrievalQueries": [
    {{"section": "0003", "query": "короткий поисковый запрос"}}
  ]
}}

OCR-текст:
{fallback_text[: self._max_input_chars]}"""

    def prepare_parts(self, text: str) -> PreparedText:
        fallback_text = self._fallback.prepare(text)
        if not fallback_text:
            self.last_retrieval_queries = []
            self.last_retrieval_decomposition = {}
            self.last_type_decision_text = ""
            return PreparedText(decision_text="", retrieval_queries=[], type_decision_text="")
        try:
            payload = parse_json_response(self._llm(self._prompt(fallback_text)))
            self.last_retrieval_decomposition = _clean_decomposition(payload)
            decision = payload.get("decisionText")
            if not isinstance(decision, str):
                # Backward-compatible parser for previous prompt contract.
                decision = payload.get("normalizedText")
            if not isinstance(decision, str):
                raise ValueError("decisionText must be a string")
            decision = compact_ocr_text(decision, self._max_output_chars)
            if len(decision) < 10:
                raise ValueError("decisionText is too short")
            type_decision = payload.get("typeDecisionText")
            if not isinstance(type_decision, str):
                type_decision = decision
            type_decision = compact_ocr_text(type_decision, self._max_output_chars)
            if len(type_decision) < 10:
                type_decision = decision
            queries = _clean_retrieval_queries(
                payload.get("retrievalQueries"), decision
            )
            self.last_retrieval_queries = queries
            self.last_type_decision_text = type_decision
            return PreparedText(
                decision_text=decision,
                retrieval_queries=queries,
                type_decision_text=type_decision,
            )
        except Exception:
            fallback = compact_ocr_text(fallback_text, self._max_output_chars)
            self.last_retrieval_queries = _fallback_query(fallback)
            self.last_retrieval_decomposition = {}
            self.last_type_decision_text = fallback
            return PreparedText(
                decision_text=fallback,
                retrieval_queries=self.last_retrieval_queries,
                type_decision_text=fallback,
            )

    def prepare(self, text: str) -> str:
        return self.prepare_parts(text).decision_text
