import json
import re
from functools import lru_cache
from pathlib import Path

FLOWER_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "floriography.json"

NEGATIVE_KEYWORDS = {
    "부정",
    "거절",
    "이별",
    "배신",
    "증오",
    "원망",
    "죽음",
    "절망",
    "복수",
    "경멸",
    "실망",
    "슬픔",
    "자만",
    "자만심",
    "오만",
    "허영",
    "질투",
    "불안",
    "고독",
    "sadness",
    "rejection",
    "betrayal",
    "hatred",
    "death",
    "despair",
    "revenge",
    "contempt",
}


def normalize_flower_key(name: str) -> str:
    return re.sub(r"\s+", " ", str(name or "").strip().lower())


def _as_meaning_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v or "").strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


@lru_cache(maxsize=1)
def _load_db() -> dict[str, dict]:
    if not FLOWER_DB_PATH.exists():
        raise FileNotFoundError(
            f"꽃말 DB를 찾을 수 없습니다: {FLOWER_DB_PATH}\n"
            "data/floriography.json 파일이 data/ 폴더에 있는지 확인해주세요."
        )

    with open(FLOWER_DB_PATH, encoding="utf-8") as f:
        raw: dict[str, list[str]] = json.load(f)

    db: dict[str, dict] = {}

    for name_en, meanings in raw.items():
        key = normalize_flower_key(name_en)
        if not key:
            continue

        db[key] = {
            "name_ko": "",
            "name_en": str(name_en).strip(),
            "meanings": _as_meaning_list(meanings),
        }

    return db


def is_negative_meaning(meaning: str) -> bool:
    text = str(meaning or "").strip().lower()
    return any(keyword in text for keyword in NEGATIVE_KEYWORDS)


def pick_local_meaning(candidates: list[str]) -> str:
    """
    Gemini 호출 실패 시 사용할 최소 fallback.
    DB 후보 중 부정적인 의미를 최대한 피해서 하나만 고르기.
    """
    clean = [m.strip() for m in candidates if str(m or "").strip()]
    if not clean:
        return ""

    for meaning in clean:
        if not is_negative_meaning(meaning):
            return meaning

    return clean[0]


def get_floriography_candidates(name_en: str) -> list[str]:
    db = _load_db()
    key = normalize_flower_key(name_en)
    entry = db.get(key)
    return list(entry.get("meanings", [])) if entry else []


def get_all_floriography_rows() -> list[dict]:
    db = _load_db()

    return [
        {
            "name_en": entry["name_en"],
            "meanings": entry["meanings"],
        }
        for entry in db.values()
    ]


def get_floriography_prompt_block() -> str:
    """
    Gemini prompt에 넣기 좋은 DB 문자열.
    추천 API에서 DB 꽃말을 우선 사용하게 하기 위해 사용.
    """
    rows = get_all_floriography_rows()

    return "\n".join(
        f"- {row['name_en']}: {', '.join(row['meanings'])}"
        for row in rows
        if row["meanings"]
    )


def map_flowers_from_yolo(yolo_names: list[str]) -> list[dict]:
    result = []

    for raw_name in yolo_names:
        name_en = str(raw_name or "").strip()
        candidates = get_floriography_candidates(name_en)

        result.append({
            "name_ko": name_en,
            "name_en": name_en,
            "meaning": pick_local_meaning(candidates),
        })

    return result


def get_flower_info(name_en: str, name_ko: str = "") -> dict:
    candidates = get_floriography_candidates(name_en)

    return {
        "name_ko": name_ko or name_en,
        "name_en": name_en,
        "meaning": pick_local_meaning(candidates),
    }