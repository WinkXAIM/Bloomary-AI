# AI 서버 API 명세

## 엔드포인트 목록

| 엔드포인트 | 메서드 | 설명 |
| --- | --- | --- |
| `/ai/detect-flowers` | POST | 꽃다발 이미지에서 꽃 종류 분류 |
| `/ai/combine-floriography` | POST | 확정된 꽃 리스트로 꽃말 조합 문구 생성 |
| `/ai/recommend-bouquet` | POST | 상황 기반 꽃다발 추천 |

> 모든 엔드포인트는 백엔드 서버 내부에서만 호출됩니다. 프론트엔드에 직접 노출되지 않습니다.

---

## POST `/ai/detect-flowers` — 꽃 종류 분류 및 분류 결과 전송

꽃다발 이미지로부터 꽃 객체 분류

**Content-Type:** `multipart/form-data`

### Request Body

| 항목 | 타입 | 필수 | 설명 |
| --- | --- | --- | --- |
| `image` | File (multipart) | 필수 | 꽃다발 이미지 |

### Response Body

| 항목 | 타입 | 설명 |
| --- | --- | --- |
| `detected_objects` | Array\<Object\> | 인식된 꽃 목록 |
| `detected_objects[].name_ko` | String | 꽃 이름(국문) |
| `detected_objects[].name_en` | String | 꽃 이름(영문) |
| `detected_objects[].meaning` | String | 꽃말(국문) |
| `detected_objects[].box2d` | Array\<Integer\> | 바운딩 박스 좌표 [x1, y1, x2, y2] |
| `detected_objects[].color_hex` | String | 바운딩 박스 내부에서 추출한 대표 꽃 색상 HEX 코드 |

### Response 예시

```json
{
  "detected_objects": [
    {
      "name_ko": "장미",
      "name_en": "Rose",
      "meaning": "불타는 사랑",
      "box2d": [100, 200, 300, 400],
      "color_hex": "#D94B5F"
    },
    {
      "name_ko": "안개꽃",
      "name_en": "Gypsophila",
      "meaning": "맑은 마음",
      "box2d": [150, 250, 350, 450],
      "color_hex": "#F2F2F2"
    }
  ]
}
```

---

## POST `/ai/combine-floriography` — 꽃말 조합 생성

사용자가 최종 확정한 꽃 리스트를 받아 감성 문구와 개별 꽃말 생성

### Request Body

| 항목 | 타입 | 필수 | 설명 |
| --- | --- | --- | --- |
| `flowers` | Array\<Object\> | 필수 | 인식된 꽃 이름 목록 |
| `flowers[].nameKo` | String | 필수 | 꽃 이름(국문) |
| `flowers[].nameEn` | String | 필수 | 꽃 이름(영문) |
| `flowers[].meaning` | String | 필수 | 꽃말(국문) |

### Response Body

| 항목 | 타입 | 설명 |
| --- | --- | --- |
| `combined_message_ko` | String | 꽃말 조합 감성 문구(국문) |
| `story_preview_en` | String | 스토리 미리보기 카드용 영문 문구 |
| `summary_en` | String | 히스토리 페이지용 짧은 요약(영문) |

### Request 예시

```json
{
  "flowers": [
    {
      "nameKo": "장미",
      "nameEn": "Rose",
      "meaning": "불타는 사랑, 열정"
    },
    {
      "nameKo": "안개꽃",
      "nameEn": "Gypsophila",
      "meaning": "맑은 마음, 사랑의 성공"
    },
    {
      "nameKo": "유칼립투스",
      "nameEn": "Eucalyptus",
      "meaning": "추억, 재생"
    }
  ]
}
```

### Response 예시

```json
{
  "combined_message_ko": "당신의 뜨거운 열정과 순수한 진심이 아름다운 추억으로 영원히 기억되길 응원합니다.",
  "story_preview_en": "A bouquet of passionate love and pure memories",
  "summary_en": "Passion & Pure Memories"
}
```

---

## POST `/ai/recommend-bouquet` — 상황별 꽃다발 추천

상황에 맞는 꽃 조합, 제목, 추천 멘트 생성

### Request Body

| 항목 | 타입 | 필수 | 설명 |
| --- | --- | --- | --- |
| `user_situation` | String | 필수 | 자연어로 입력한 상황 설명 |

### Response Body

| 항목 | 타입 | 설명 |
| --- | --- | --- |
| `title` | String | 추천 꽃다발 제목 |
| `combined_message` | String | 꽃다발 꽃말 조합 문구 |
| `flowers` | Array\<Object\> | 추천 꽃 상세 목록 |
| `flowers[].name_ko` | String | 꽃 이름(국문) |
| `flowers[].name_en` | String | 꽃 이름(영문) |
| `flowers[].meaning` | String | 꽃말(국문) |

### Request 예시

```json
{
  "user_situation": "프로포즈 꽃다발 추천 ㄱ"
}
```

### Response 예시

```json
{
  "title": "익숙함 속에 피어난 따뜻한 진심",
  "combined_message": "오랜 시간 곁을 지켜준 소중한 당신에게, 변치 않는 우정과 따스한 위로를 담아 고마움을 전합니다.",
  "flowers": [
    {
      "name_ko": "메리골드",
      "name_en": "Marigold",
      "meaning": "반드시 오고야 말 행복"
    },
    {
      "name_ko": "프리지아",
      "name_en": "Freesia",
      "meaning": "당신의 시작을 응원합니다, 천진난만"
    },
    {
      "name_ko": "유칼립투스",
      "name_en": "Eucalyptus",
      "meaning": "추억, 재생, 위로"
    }
  ]
}
```

---

## 에러 코드

| HTTP 상태코드 | 에러코드 | 설명 |
| --- | --- | --- |
| 400 | `INVALID_REQUEST` | 잘못된 요청 파라미터 |
| 422 | `AI_PROCESSING_ERROR` | AI 모델 처리 실패 |
| 500 | `INTERNAL_SERVER_ERROR` | 서버 내부 오류 |
