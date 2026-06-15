# Bloomary-AI

## Directory Structure

* **data/**: 프로젝트에 필요한 데이터 보관

  * `floriography.json`: 꽃 이름별 꽃말 데이터

* **models/**: 학습된 YOLO 모델 가중치 보관

  * 모델 파일은 용량 및 관리 문제로 git에 포함하지 않습니다.
  * 로컬 실행 시 아래 경로에 직접 추가해주세요.

    ```text
    models/single_flower_yolo11m_v1_1024_fitblack_3x_best.pt
    ```

* **src/server/**: FastAPI 서버 및 API 관련 소스 코드

* **src/logic/**: 꽃 탐지, 꽃말 조합, 추천 로직 코드

* **notebooks/**: 학습 실험 및 데이터 분석 기록

* **Dockerfile**: 서버 배포를 위한 Docker 설정 파일

* **test/**: 로컬 테스트용 코드 및 이미지

  * 테스트 이미지는 git에 포함하지 않습니다.

* **requirements.txt**: 로컬 및 Docker 실행에 필요한 Python 패키지 목록

* **.env.example**: 환경변수 예시 파일

---

## Branch Strategy

1. **main**: 최종 배포 브랜치
2. **feature/이슈번호-기능명** 또는 **chore/작업명**: 각자 맡은 기능이나 설정 작업을 개발하는 작업 공간

   * 예: `feature/1-yolo-fastapi`, `feature/2-llm-prompt`
   * 작업 완료 시 `main` 브랜치로 Pull Request(PR)를 생성합니다.

---

## Gemini Models

* `detector_single_flower.py`: `.env`의 `GEMINI_MODEL_NAME` 값을 사용합니다.

  * 값이 없으면 코드의 기본 모델명을 사용합니다.
* `llm_handler.py`: 꽃말 조합 / 꽃다발 추천용 Gemini 모델을 사용합니다.

Gemini API quota 제한이 있을 수 있으므로 이미지 검수용 모델과 텍스트 생성용 모델을 분리해서 사용합니다.

---

## Setup

### 1. 가상환경 생성 및 활성화

Python venv를 사용하는 경우:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows 환경에서는 아래 명령어를 사용합니다.

```bash
.venv\Scripts\activate
```

Conda 환경을 사용하는 경우 기존 환경을 활성화해도 됩니다.

```bash
conda activate tftenv
```

---

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

---

### 3. 환경변수 설정

`.env.example` 파일을 복사해서 `.env` 파일을 생성합니다.

```bash
cp .env.example .env
```

Windows PowerShell에서는 아래 명령어를 사용할 수 있습니다.

```bash
copy .env.example .env
```

`.env` 파일에 필요한 환경변수를 입력합니다.

```env
GEMINI_API_KEY=your_actual_gemini_api_key
GEMINI_MODEL_NAME=models/gemini-3-flash-preview

YOLO_MODEL_PATH=models/single_flower_yolo11m_v1_1024_fitblack_3x_best.pt
YOLO_DEVICE=cpu
YOLO_IMG_SIZE=1024
YOLO_CONF=0.25
YOLO_IOU=0.45
YOLO_MAX_DET=80

GEMINI_MAX_CANDIDATES=28
```

---

### 4. YOLO 모델 파일 추가

YOLO 모델 가중치 파일은 git에 포함하지 않습니다.

아래 경로에 모델 파일을 직접 추가해주세요.

```text
models/single_flower_yolo11m_v1_1024_fitblack_3x_best.pt
```

---

### 5. 로컬 테스트 이미지 추가

테스트 이미지를 아래 경로에 추가합니다.

```text
test/test.jpeg
```

테스트 이미지는 git에 커밋하지 않습니다.

---

### 6. 로컬 테스트 실행

```bash
python test/test_single_flower.py
```

실행 결과는 터미널에 JSON 형태로 출력됩니다.

또한 바운딩박스 확인용 이미지가 아래 경로에 저장됩니다.

```text
test/annotated_result.jpg
```

`annotated_result.jpg`에서는 탐지된 꽃의 바운딩박스와 대표 색상 HEX 값을 확인할 수 있습니다.

예상 응답 형태:

```json
{
  "detected_objects": [
    {
      "name_ko": "장미",
      "name_en": "rose",
      "meaning": "사랑",
      "box2d": [100, 200, 300, 400],
      "color_hex": "#D94B5F"
    }
  ]
}
```

---

## FastAPI Server

### 1. 로컬 서버 실행

```bash
uvicorn src.server.main:app --host 0.0.0.0 --port 8000 --reload
```

서버 실행 후 Swagger 문서는 아래 주소에서 확인할 수 있습니다.

```text
http://127.0.0.1:8000/docs
```

Health check는 아래 주소에서 확인할 수 있습니다.

```text
http://127.0.0.1:8000/health
```

---

### 2. 꽃 탐지 API 테스트

```bash
curl -X POST "http://127.0.0.1:8000/ai/detect-flowers" \
  -F "image=@test/test.jpeg"
```

---

### 3. 꽃말 조합 API 테스트

```bash
curl -X POST "http://127.0.0.1:8000/ai/combine-floriography" \
  -H "Content-Type: application/json" \
  -d '{
    "flowers": [
      {
        "nameKo": "장미",
        "nameEn": "rose",
        "meaning": "사랑"
      },
      {
        "nameKo": "튤립",
        "nameEn": "tulip",
        "meaning": "고백"
      }
    ]
  }'
```

---

### 4. 꽃다발 추천 API 테스트

```bash
curl -X POST "http://127.0.0.1:8000/ai/recommend-bouquet" \
  -H "Content-Type: application/json" \
  -d '{
    "user_situation": "친구 생일에 줄 밝고 따뜻한 느낌의 꽃다발을 추천해줘"
  }'
```

---

## Docker

### 1. Docker 이미지 빌드

```bash
docker build -t bloomary-ai .
```

---

### 2. Docker 컨테이너 실행

모델 파일은 Docker 이미지에 포함하지 않고, 로컬 `models/` 폴더를 컨테이너에 mount합니다.

`.env` 파일을 사용하는 경우:

```bash
docker run --env-file .env \
  -v "$(pwd)/models:/app/models:ro" \
  -p 8000:8000 \
  bloomary-ai
```

Windows PowerShell에서는 아래 명령어를 사용할 수 있습니다.

```powershell
docker run --env-file .env `
  -v "${PWD}/models:/app/models:ro" `
  -p 8000:8000 `
  bloomary-ai
```

실행 후 Swagger 문서는 아래 주소에서 확인할 수 있습니다.

```text
http://127.0.0.1:8000/docs
```

Health check는 아래 주소에서 확인할 수 있습니다.

```text
http://127.0.0.1:8000/health
```

---

### 3. Docker 실행 시 필요한 파일

Docker 실행 전에 아래 파일들이 로컬에 있어야 합니다.

```text
models/single_flower_yolo11m_v1_1024_fitblack_3x_best.pt
data/floriography.json
.env
```

모델 파일과 `.env`는 git에 포함하지 않으므로, 로컬 또는 배포 환경에서 직접 준비해야 합니다.

---

## Model

현재 사용 모델:

```text
YOLO11m
task: single-flower candidate detection
input image size: 1024
model file: models/single_flower_yolo11m_v1_1024_fitblack_3x_best.pt
```

추론 설정:

```text
imgsz=1024
conf=0.25
iou=0.45
agnostic_nms=True
max_det=80
device=cpu
```

처리 흐름:

```text
1. YOLO11m detects individual flower candidates.
2. Candidate boxes are filtered and deduplicated.
3. Candidate crops are arranged into a crop grid.
4. Gemini classifies flower names from the crop candidates.
5. The server extracts representative color_hex values and returns the final API response.
```

같은 실제 객체로 보이는 중복 bbox는 서버 내부에서 제거합니다.
또한 같은 꽃 이름이 여러 번 나온 경우, 현재 로직에서는 대표 객체 하나만 남깁니다.

---

## API Response Example

### `/ai/detect-flowers`

```json
{
  "detected_objects": [
    {
      "name_ko": "장미",
      "name_en": "rose",
      "meaning": "사랑",
      "box2d": [100, 200, 300, 400],
      "color_hex": "#D94B5F"
    }
  ]
}
```

### `/ai/combine-floriography`

```json
{
  "combined_message_ko": "사랑과 고백의 마음을 담아 진심을 전합니다.",
  "story_preview_en": "A heartfelt bouquet of love and confession",
  "summary_en": "Love & Confession"
}
```

### `/ai/recommend-bouquet`

```json
{
  "title": "따뜻한 생일 축하의 꽃다발",
  "combined_message": "밝은 기쁨과 따뜻한 마음을 담아 소중한 생일을 축하합니다.",
  "flowers": [
    {
      "name_ko": "장미",
      "name_en": "Rose",
      "meaning": "사랑"
    },
    {
      "name_ko": "프리지아",
      "name_en": "Freesia",
      "meaning": "응원"
    }
  ]
}
```

---

## Error Code

| HTTP Status Code | Error Code              | Description |
| ---------------- | ----------------------- | ----------- |
| 400              | `INVALID_REQUEST`       | 잘못된 요청 파라미터 |
| 422              | `AI_PROCESSING_ERROR`   | AI 모델 처리 실패 |
| 500              | `INTERNAL_SERVER_ERROR` | 서버 내부 오류    |

---

## Git Ignore Policy

아래 파일들은 git에 포함하지 않습니다.

```text
.env
models/*.pt
models/*.onnx
models/*.engine
test/*.jpeg
test/*.jpg
test/*.png
test/annotated_result.jpg
__pycache__/
*.pyc
```