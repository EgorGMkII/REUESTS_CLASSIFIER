from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
INPUT_DATASET = ROOT / "data" / "evaluation" / "real_v3.json"
OUTPUT_DATASET = ROOT / "data" / "evaluation" / "real_plus_synth.json"
THEMES_PATH = ROOT / "data" / "classifiers" / "themes_leaf.json"
SUBTYPES_PATH = ROOT / "data" / "classifiers" / "question_subtypes.json"
OCR_RUNS_DIR = ROOT / "data" / "evaluation" / "hydra_ocr_runs"


QUESTION_TYPES = {
    "1": "Предложение",
    "2": "Заявление",
    "3": "Жалоба",
    "4": "Не обращение",
    "5": "Запрос информации",
    "6": "Коллективное обращение",
    "7": "Служебный запрос",
    "8": "Дубликат",
    "9": "Иное",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def catalog_by_code(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    items = payload["items"] if isinstance(payload, dict) and "items" in payload else payload
    return {item["code"]: item for item in items}


def compact_ocr_text(text: str) -> str:
    """Keep the first extracted appeal when OCR accidentally appended an agency response."""

    marker = "\nКраткое тело обращения:"
    if marker not in text:
        return text.strip()
    first, _rest = text.split(marker, 1)
    return first.strip()


def expected_label(
    theme_catalog: dict[str, dict[str, Any]],
    subtype_catalog: dict[str, dict[str, Any]],
    theme_codes: list[str],
    question_type_code: str,
    question_subtype_code: str,
) -> dict[str, Any]:
    subtype = subtype_catalog[question_subtype_code]
    return {
        "themes": [
            {
                "code": code,
                "name": theme_catalog[code]["name"],
                "section": theme_catalog[code]["section"],
            }
            for code in theme_codes
        ],
        "questionType": {
            "code": question_type_code,
            "name": QUESTION_TYPES[question_type_code],
        },
        "questionSubtype": {
            "code": subtype["code"],
            "officialCode": subtype["officialCode"],
            "name": subtype["name"],
        },
    }


def real_case(
    case_id: str,
    pdf_number: int,
    theme_catalog: dict[str, dict[str, Any]],
    subtype_catalog: dict[str, dict[str, Any]],
    theme_codes: list[str],
    question_type_code: str,
    question_subtype_code: str,
    notes: str,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    run_dir = OCR_RUNS_DIR / case_id
    text_path = run_dir / "text.txt"
    debug_path = run_dir / "hydra_ocr_debug.json"
    text = compact_ocr_text(text_path.read_text(encoding="utf-8"))
    return {
        "id": case_id,
        "text": text,
        "expected": expected_label(
            theme_catalog,
            subtype_catalog,
            theme_codes,
            question_type_code,
            question_subtype_code,
        ),
        "tags": [
            "real",
            "hydra-ocr",
            "real-plus",
            "human-inferred-label",
            *(tags or []),
        ],
        "source": {
            "pdf": f"Обращения\\{pdf_number}.pdf",
            "hydraOcrText": str(text_path.relative_to(ROOT)),
            "hydraOcrDebug": str(debug_path.relative_to(ROOT)),
            "labelNotes": notes,
        },
    }


def synth_case(
    case_id: str,
    text: str,
    theme_catalog: dict[str, dict[str, Any]],
    subtype_catalog: dict[str, dict[str, Any]],
    theme_codes: list[str],
    question_type_code: str,
    question_subtype_code: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "text": text.strip(),
        "expected": expected_label(
            theme_catalog,
            subtype_catalog,
            theme_codes,
            question_type_code,
            question_subtype_code,
        ),
        "tags": ["synthetic", "real-plus"],
        "source": {
            "kind": "synthetic",
            "labelNotes": notes,
        },
    }


def build_real_cases(
    theme_catalog: dict[str, dict[str, Any]],
    subtype_catalog: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        real_case(
            "real-016",
            16,
            theme_catalog,
            subtype_catalog,
            ["0002.0013.0139.0338", "0002.0014.0144.0440"],
            "2",
            "2.1.3",
            "Заявитель просит сохранить детскую хоккейную секцию и доступность занятий для детей.",
        ),
        real_case(
            "real-018",
            18,
            theme_catalog,
            subtype_catalog,
            ["0001.0002.0027.0124", "0005.0005.0056.1163"],
            "3",
            "3.2.1",
            "Есть прямое обжалование действий должностных лиц при рассмотрении обращения; предметно затронут расчет жилищно-коммунальной субсидии.",
            ["needs-review"],
        ),
        real_case(
            "real-019",
            19,
            theme_catalog,
            subtype_catalog,
            ["0004.0015.0158.0970", "0002.0013.0141.0370"],
            "3",
            "3.2.1",
            "Заявитель жалуется на бездействие органов по воинскому захоронению/мемориалу; OCR дополнительно извлек ответ ведомства, он отсечен.",
            ["needs-review", "ocr-response-trimmed"],
        ),
        real_case(
            "real-020",
            20,
            theme_catalog,
            subtype_catalog,
            ["0005.0005.0056.1154", "0005.0005.0056.1169"],
            "2",
            "2.2.2",
            "Сообщение о систематическом нарушении сроков восстановления горячего водоснабжения и просьба принять меры.",
        ),
    ]


def build_synthetic_cases(
    theme_catalog: dict[str, dict[str, Any]],
    subtype_catalog: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        synth_case(
            "synth-001",
            """
            Работники подрядной организации сообщают, что за два месяца им не выплачена заработная плата.
            Руководство обещает перечислить деньги после приемки работ, но конкретных сроков не называет.
            Просим провести проверку, защитить право работников на оплату труда и обязать работодателя погасить задолженность.
            """,
            theme_catalog,
            subtype_catalog,
            ["0002.0006.0065.0257"],
            "3",
            "3.1.1",
            "Явное требование защитить нарушенное право на оплату труда.",
        ),
        synth_case(
            "synth-002",
            """
            Автобусы маршрута № 42 регулярно проходят мимо остановки «Северная» без посадки пассажиров.
            Из-за этого жители опаздывают на работу и в поликлинику, особенно утром.
            Просим проверить работу перевозчика и обеспечить нормальное транспортное обслуживание.
            """,
            theme_catalog,
            subtype_catalog,
            ["0003.0009.0099.0733"],
            "2",
            "2.1.1",
            "Просьба решить конкретную проблему пассажирского обслуживания.",
        ),
        synth_case(
            "synth-003",
            """
            Прошу сообщить, какие документы нужны для получения государственной услуги через МФЦ,
            можно ли подать заявление дистанционно и в какие часы ведется прием граждан.
            Других требований или жалоб не заявляю.
            """,
            theme_catalog,
            subtype_catalog,
            ["0001.0002.0025.0092"],
            "5",
            "-",
            "Самостоятельный запрос сведений о порядке получения услуги.",
        ),
        synth_case(
            "synth-004",
            """
            Предлагаю организовать в районе сеть контейнерных площадок для раздельного сбора отходов
            и разместить понятные инструкции для жителей. Это позволит сократить объем мусора,
            направляемого на полигон, и улучшить экологическую ситуацию.
            """,
            theme_catalog,
            subtype_catalog,
            ["0005.0005.0056.1160", "0003.0011.0122.0839"],
            "1",
            "1.3.4",
            "Общая рекомендация по улучшению обращения с отходами.",
        ),
        synth_case(
            "synth-005",
            """
            Я получаю льготное лекарство по рецепту, но в аптеке его нет уже третий месяц.
            Врач замену не назначает, состояние ухудшается.
            Прошу помочь обеспечить выдачу назначенного препарата и проверить работу ответственных организаций.
            """,
            theme_catalog,
            subtype_catalog,
            ["0002.0014.0143.0420"],
            "2",
            "2.1.1",
            "Просьба о содействии в лекарственном обеспечении.",
        ),
        synth_case(
            "synth-006",
            """
            Родители учеников сообщают, что в здании школы несколько недель низкая температура в классах.
            Дети сидят на уроках в верхней одежде, занятия часто сокращают.
            Просим проверить условия образовательного процесса и обеспечить нормальное теплоснабжение школы.
            """,
            theme_catalog,
            subtype_catalog,
            ["0002.0013.0139.0332", "0002.0013.0139.0333"],
            "2",
            "2.1.3",
            "Проблема условий обучения и теплоснабжения касается прав детей.",
        ),
        synth_case(
            "synth-007",
            """
            Рядом с жилыми домами образовалась несанкционированная свалка строительного мусора и бытовых отходов.
            Отходы разносит ветром, появился запах, жители опасаются загрязнения почвы.
            Просим провести проверку, убрать свалку и установить ответственных лиц.
            """,
            theme_catalog,
            subtype_catalog,
            ["0005.0005.0056.1161", "0003.0011.0122.0834"],
            "2",
            "2.2.2",
            "Сообщение о нарушении экологических/санитарных требований и просьба принять меры.",
        ),
        synth_case(
            "synth-008",
            """
            В сельском поселении несколько хозяйств столкнулись с падежом крупного рогатого скота.
            Ветеринарная служба выезжает с задержкой, анализы не берут, рекомендации владельцам не дают.
            Просим проверить работу ветеринарной службы и принять меры для предотвращения распространения заболевания.
            """,
            theme_catalog,
            subtype_catalog,
            ["0003.0009.0098.0717", "0003.0009.0098.0725"],
            "2",
            "2.2.3",
            "Сообщение о недостатках работы ветеринарной службы и проблеме животноводства.",
        ),
        synth_case(
            "synth-009",
            """
            При уточнении границ земельного участка возник спор с соседним землепользователем.
            Администрация отказывает в содействии и не разъясняет порядок согласования границ.
            Прошу помочь защитить право на земельный участок и организовать рассмотрение земельного спора.
            """,
            theme_catalog,
            subtype_catalog,
            ["0003.0011.0123.0845"],
            "2",
            "2.1.1",
            "Просьба о содействии в защите права на землю.",
        ),
        synth_case(
            "synth-010",
            """
            Прошу отменить постановление о привлечении меня к административной ответственности,
            так как протокол составлен без моего участия, доказательства нарушения не представлены,
            а мои объяснения не были рассмотрены должностным лицом.
            """,
            theme_catalog,
            subtype_catalog,
            ["0001.0002.0028.0159", "0004.0016.0162.1007"],
            "3",
            "3.2.4",
            "Прямое обжалование привлечения к административной ответственности.",
        ),
    ]


def main() -> int:
    base_dataset = load_json(INPUT_DATASET)
    theme_catalog = catalog_by_code(THEMES_PATH)
    subtype_catalog = catalog_by_code(SUBTYPES_PATH)

    cases = list(base_dataset["cases"])
    cases.extend(build_real_cases(theme_catalog, subtype_catalog))
    cases.extend(build_synthetic_cases(theme_catalog, subtype_catalog))

    output = {
        "datasetVersion": "plus-synth-1.0",
        "classifierVersion": base_dataset.get("classifierVersion", "2025-10-31"),
        "cases": cases,
        "meta": {
            "baseDataset": str(INPUT_DATASET.relative_to(ROOT)),
            "baseCases": len(base_dataset["cases"]),
            "addedRealInferredCases": 4,
            "addedSyntheticCases": 10,
            "notes": [
                "real-016, real-018, real-019, real-020 are inferred labels from Hydra OCR text and should be reviewed before treating as gold labels.",
                "real-019 OCR contained an agency response after the appeal; only the first extracted appeal block is used.",
                "Synthetic cases are intended for diagnostic coverage of common classifier boundaries, not as official ground truth.",
            ],
        },
    }
    save_json(OUTPUT_DATASET, output)
    print(f"Saved {OUTPUT_DATASET}")
    print(f"Total cases: {len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
