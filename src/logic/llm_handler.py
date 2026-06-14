import json
import os
import re
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

from src.logic.flower_names import FLOWER_NAME_KO_MAP, BOUQUET_RECOMMENDABLE_FLOWER_KEYS

from src.logic.flower_mapper import (
    get_floriography_candidates,
    pick_local_meaning,
    is_negative_meaning,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

if not api_key:
    raise RuntimeError(
        "Gemini API key가 없습니다. .env에 GEMINI_API_KEY를 설정하거나 환경변수로 export 해주세요."
    )

genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-3.1-flash-lite")


def _clean_json_text(text: str) -> str:
    text = str(text or "").strip()

    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()
    elif text.startswith("```"):
        text = text.removeprefix("```").strip()

    if text.endswith("```"):
        text = text.removesuffix("```").strip()

    return text


def _parse_json_object(text: str) -> dict:
    cleaned = _clean_json_text(text)

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"Gemini 응답에서 JSON 객체를 찾지 못했습니다: {text}")

    data = json.loads(match.group(0))

    if not isinstance(data, dict):
        raise ValueError("Gemini 응답 JSON이 객체 형식이 아닙니다.")

    return data


def _generate_json(prompt: str) -> dict:
    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"},
    )
    return _parse_json_object(response.text)


def _fallback_detect_meaning(flower: dict) -> str:
    candidates = get_floriography_candidates(flower.get("name_en", ""))
    return pick_local_meaning(candidates)


def attach_contextual_meanings(detected_objects: list[dict]) -> list[dict]:
    """
    /ai/detect-flowers 결과의 detected_objects[].meaning을 채운다.

    - DB 후보가 있으면 Gemini가 DB 후보를 우선해서 한국어 1~2개로 정제
    - DB 후보가 비어 있으면 Gemini가 직접 일반 꽃말을 찾아 작성
    - Gemini 호출 자체가 실패하면 에러를 숨기지 않고 그대로 올림
    """
    if not detected_objects:
        return detected_objects

    flowers_for_prompt = []

    for index, flower in enumerate(detected_objects):
        name_en = str(flower.get("name_en", "")).strip()
        name_ko = str(flower.get("name_ko", "")).strip()
        candidates = get_floriography_candidates(name_en)

        flowers_for_prompt.append({
            "index": index,
            "name_ko": name_ko,
            "name_en": name_en,
            "db_meaning_candidates": candidates,
        })

    prompt = f"""
아래는 꽃다발 이미지에서 탐지된 꽃 목록입니다.
각 꽃의 최종 꽃말 meaning을 한국어로 골라주세요.

절대 규칙:
1. db_meaning_candidates가 비어 있지 않은 꽃은 반드시 그 후보 안에서만 meaning을 선택하세요.
2. db_meaning_candidates에 영어 후보가 있으면 한국어로 번역/의역할 수 있지만, 의미를 새로 추가하면 안 됩니다.
   예: "Prosperity" -> "번영", "good fortune" -> "행운"
3. db_meaning_candidates가 있는 꽃에는 모델의 일반 지식이나 검색 지식을 사용하지 마세요.
4. db_meaning_candidates가 비어 있는 꽃만 일반적으로 알려진 꽃말을 찾아 meaning을 작성하세요.
5. 각 meaning은 한국어로 1개를 권장하고, 꼭 필요할 때만 쉼표로 구분해 최대 2개까지 허용합니다.
6. 꽃다발에 여러 꽃이 있으면 전체 조합과 분위기를 고려해서 후보 중 가장 잘 어울리는 것을 고르세요.
7. 부정적인 꽃말은 가능한 제외하세요.
   단, 후보가 부정적인 의미뿐이면 가장 덜 부정적인 표현으로 순화하세요.
8. 같은 꽃이 여러 번 나오면 같은 meaning을 사용하세요.
9. name_ko, name_en, box2d, color_hex는 수정하지 않습니다.
10. 설명 없이 JSON만 응답하세요.

좋은 예:
- rose 후보가 ["열렬한 사랑", "질투", "순결", "Love", "passion", "beauty"]이면
  꽃다발 맥락에 따라 "열렬한 사랑" 또는 "순결" 또는 "사랑"처럼 후보 기반으로만 선택합니다.
- peony 후보가 ["수줍음", "Prosperity", "good fortune"]이면
  "수줍음", "번영", "행운" 중에서만 선택합니다.
  "행복한 결혼"처럼 후보에 없는 꽃말은 쓰면 안 됩니다.
- gypsophila 후보가 []이면
  일반적으로 알려진 꽃말을 찾아 "맑은 마음", "순수한 사랑"처럼 작성할 수 있습니다.

탐지된 꽃 목록:
{json.dumps(flowers_for_prompt, ensure_ascii=False)}

응답 형식:
{{
  "detected_objects": [
    {{
      "index": 0,
      "meaning": "한국어 꽃말"
    }}
  ]
}}
"""

    data = _generate_json(prompt)
    meaning_items = data.get("detected_objects", [])

    meaning_by_index: dict[int, str] = {}

    for item in meaning_items:
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue

        meaning = str(item.get("meaning", "")).strip()
        if meaning:
            meaning_by_index[index] = meaning

    result = []

    for index, flower in enumerate(detected_objects):
        copied = dict(flower)
        copied["meaning"] = meaning_by_index.get(index) or _fallback_detect_meaning(copied)
        result.append(copied)

    return result


def combine_floriography(flowers: list[dict]) -> dict:
    flower_list = "\n".join(
        f"- {f['name_ko']}({f['name_en']}): {f['meaning']}" for f in flowers
    )

    prompt = f"""
아래 꽃들의 개별 꽃말을 조합해서 자연스럽고 감성적인 문구를 만들어주세요.
반드시 아래 JSON 형식으로만 응답하세요. 다른 설명은 쓰지 마세요.

꽃 목록:
{flower_list}

절대 규칙:
1. combined_message_ko는 한국어로 작성하세요.
2. combined_message_ko는 입력된 meaning만 사용해서 자연스럽게 조합하세요.
3. 입력에 없는 새로운 꽃말을 추가하지 마세요.
4. combined_message_ko는 1문장으로 작성하세요.
5. story_preview_en는 combined_message_ko를 감성적으로 영어로 옮긴 짧은 문구입니다.
6. story_preview_en는 5~10단어 정도로 작성하세요.
7. summary_en는 story_preview_en보다 더 짧게 2~5단어로 작성하세요.
8. 부정적인 표현은 가능한 피하고, 따뜻하고 감성적인 톤으로 작성하세요.

좋은 예:
입력:
- 장미: 열렬한 사랑
- 안개꽃: 순수한 마음
출력:
{{
  "combined_message_ko": "열렬한 사랑을 순수한 마음으로 전합니다.",
  "story_preview_en": "Passionate love, delivered with purity.",
  "summary_en": "Passionate Love"
}}

응답 형식:
{{
  "combined_message_ko": "꽃말을 조합한 한국어 감성 문구",
  "story_preview_en": "Short English story preview",
  "summary_en": "Short English summary"
}}
"""

    return _generate_json(prompt)

def _flower_key_to_name_en(key: str) -> str:
    return str(key or "").replace("_", " ").title()


def _normalize_meaning_text(text: str) -> str:
    return str(text or "").strip().lower().replace(" ", "").replace(",", "")

MEANING_TRANSLATION_MAP = {
    "love": "사랑",
    "passion": "열정",
    "beauty": "아름다움",
    "prosperity": "번영",
    "goodfortune": "행운",
    "devotion": "헌신",
    "strength": "강인함",
    "purity": "순수",
    "innocence": "순수",
    "confidence": "자신감",
    "mutualsupport": "상호 지지",
    "admiration": "존경",
    "friendship": "우정",
    "gratitude": "감사",
    "happiness": "행복",
    "joy": "기쁨",
    "hope": "희망",
}

def _is_allowed_db_meaning(generated: str, candidates: list[str]) -> bool:
    if not candidates:
        return True

    generated_norm = _normalize_meaning_text(generated)

    for candidate in candidates:
        candidate_norm = _normalize_meaning_text(candidate)

        if candidate_norm and candidate_norm in generated_norm:
            return True

    for candidate in candidates:
      key = _normalize_meaning_text(candidate)
      translated = MEANING_TRANSLATION_MAP.get(key)

      if translated and translated in generated:
          return True

    return False

def _translate_known_meaning_to_ko(meaning: str) -> str:
    text = str(meaning or "").strip()
    key = _normalize_meaning_text(text)
    return MEANING_TRANSLATION_MAP.get(key, text)

def _pick_positive_local_meaning(candidates: list[str]) -> str:
    clean = [str(m or "").strip() for m in candidates if str(m or "").strip()]

    if not clean:
        return ""

    for meaning in clean:
        if not is_negative_meaning(meaning):
            return meaning

    return clean[0]

def _build_recommendable_flower_rows() -> list[dict]:
    rows = []

    for flower_key in sorted(BOUQUET_RECOMMENDABLE_FLOWER_KEYS):
        if flower_key not in FLOWER_NAME_KO_MAP:
            continue

        rows.append({
            "flower_key": flower_key,
            "name_ko": FLOWER_NAME_KO_MAP[flower_key],
            "name_en": _flower_key_to_name_en(flower_key),
            "meaning_candidates": get_floriography_candidates(flower_key),
        })

    return rows


def _finalize_recommended_flowers(raw_flowers: list[dict]) -> list[dict]:
    finalized = []
    used_keys = set()

    for item in raw_flowers:
        flower_key = str(item.get("flower_key", "")).strip().lower()

        if flower_key in used_keys:
            continue

        if flower_key not in BOUQUET_RECOMMENDABLE_FLOWER_KEYS:
            continue

        if flower_key not in FLOWER_NAME_KO_MAP:
            continue

        candidates = get_floriography_candidates(flower_key)
        generated_meaning = str(item.get("meaning", "")).strip()

        if candidates and not _is_allowed_db_meaning(generated_meaning, candidates):
            meaning = _pick_positive_local_meaning(candidates)
        else:
            meaning = generated_meaning or _pick_positive_local_meaning(candidates)

        if candidates and is_negative_meaning(meaning):
            alternative = _pick_positive_local_meaning(candidates)

            if alternative and not is_negative_meaning(alternative):
                meaning = alternative

        meaning = _translate_known_meaning_to_ko(meaning)

        if not meaning:
            continue

        finalized.append({
            "name_ko": FLOWER_NAME_KO_MAP[flower_key],
            "name_en": _flower_key_to_name_en(flower_key),
            "meaning": meaning,
        })

        used_keys.add(flower_key)

    return finalized


def _fallback_recommend_flowers() -> list[dict]:
    fallback_keys = ["rose", "tulip", "gypsophila"]

    flowers = []

    for flower_key in fallback_keys:
        candidates = get_floriography_candidates(flower_key)
        meaning = pick_local_meaning(candidates)

        if not meaning and flower_key == "gypsophila":
            meaning = "맑은 마음"

        flowers.append({
            "name_ko": FLOWER_NAME_KO_MAP[flower_key],
            "name_en": _flower_key_to_name_en(flower_key),
            "meaning": meaning,
        })

    return flowers

def recommend_bouquet(user_situation: str) -> dict:
    recommendable_flowers = _build_recommendable_flower_rows()

    prompt = f"""
사용자의 상황에 어울리는 꽃다발을 추천해주세요.
반드시 JSON 형식으로만 응답하세요. 다른 설명은 쓰지 마세요.

사용자 상황:
{user_situation}

추천 가능한 꽃 후보:
{json.dumps(recommendable_flowers, ensure_ascii=False)}

절대 규칙:
1. flowers는 2~4개로 구성하세요.
2. flowers[].flower_key는 반드시 추천 가능한 꽃 후보의 flower_key 중에서만 고르세요.
3. name_ko, name_en은 절대 직접 만들지 마세요.
4. meaning_candidates가 비어 있지 않은 꽃은 반드시 그 후보 안에서만 meaning을 선택하세요.
5. meaning_candidates의 영어 후보는 한국어로 번역/의역할 수 있지만, 후보에 없는 꽃말을 새로 만들면 안 됩니다.
6. meaning_candidates가 비어 있는 꽃은 사용자의 상황에 꼭 맞을 때만 선택하세요.
7. 꽃다발에 흔히 쓰이는 꽃 위주로 추천하세요.
8. 부정적이거나 상황에 어울리지 않는 꽃말은 제외하세요. 예: 자만심, 질투, 거절, 이별, 절망, 배신
9. 같은 의미의 꽃만 반복하지 말고, 상황에 맞는 의미 조합이 되도록 고르세요.
10. combined_message는 선택한 flowers[].meaning을 실제로 조합해서 자연스럽고 감성적인 한국어 문장으로 작성하세요.
11. title은 사용자의 상황에 맞는 한국어 제목으로 작성하세요.
12. 설명 없이 JSON만 응답하세요.

응답 형식:
{{
  "title": "추천 꽃다발 제목",
  "combined_message": "꽃다발 전체 꽃말 조합 문구",
  "flowers": [
    {{
      "flower_key": "rose",
      "meaning": "열렬한 사랑"
    }}
  ]
}}
"""

    data = _generate_json(prompt)

    finalized_flowers = _finalize_recommended_flowers(data.get("flowers", []))

    if len(finalized_flowers) < 2:
        finalized_flowers = _fallback_recommend_flowers()

    combined_message = str(data.get("combined_message", "")).strip()

    if not combined_message:
        meanings = [flower["meaning"] for flower in finalized_flowers if flower.get("meaning")]
        combined_message = f"{'과 '.join(meanings[:2])}을 담아 마음을 전합니다."

    return {
        "title": str(data.get("title", "")).strip() or "추천 꽃다발",
        "combined_message": combined_message,
        "flowers": finalized_flowers,
    }