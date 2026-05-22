# Bloomary-AI

## Directory Structure

- **data/**: 프로젝트에 필요한 데이터 보관
  - `floriography.json`: 꽃 이름별 꽃말 데이터
- **models/**: 학습된 YOLO 모델 가중치 보관
  - 모델 파일은 용량 및 관리 문제로 git에 포함하지 않습니다.
  - 로컬 실행 시 아래 경로에 직접 추가해주세요.

    ```text
    models/bouquet_yolo11s_v3_704_best.pt
    ```

- **src/server/**: FastAPI 서버 및 API 관련 소스 코드
- **src/logic/**: 꽃 탐지, 꽃말 조합, 추천 로직 코드
- **notebooks/**: 학습 실험 및 데이터 분석 기록
- **Dockerfile**: 서버 배포를 위한 Docker 설정 파일
- **test/**: 로컬 테스트용 코드 및 이미지
  - 테스트 이미지는 git에 포함하지 않습니다.
- **requirements.txt**: 로컬 및 Docker 실행에 필요한 Python 패키지 목록
- **.env.example**: 환경변수 예시 파일

---

## Branch Strategy

1. **main**: 최종 배포 브랜치
2. **feature/이슈번호-기능명**: 각자 맡은 기능을 개발하는 작업 공간
   - 예: `feature/1-yolo-fastapi`, `feature/2-llm-prompt`
   - 작업 완료 시 `main` 브랜치로 Pull Request(PR)를 생성합니다.

---

## Gemini Models

Gemini 모델명은 코드에서 직접 설정합니다.

- `detector.py`: 꽃 이미지 검수용 Gemini 모델
- `llm_handler.py`: 꽃말 조합 / 꽃다발 추천용 Gemini 모델

개발 중 Gemini API quota 제한이 있을 수 있으므로, 필요하면 코드에서 텍스트용 모델과 이미지 검수용 모델을 분리해서 사용합니다.

## Setup

### 1. 가상환경 생성 및 활성화

Python venv를 사용하는 경우:

```bash
python -m venv .venv
source .venv/bin/activate
````

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
```

---

### 4. YOLO 모델 파일 추가

YOLO 모델 가중치 파일은 git에 포함하지 않습니다.

아래 경로에 모델 파일을 직접 추가해주세요.

```text
models/bouquet_yolo11s_v3_704_best.pt
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
python test/test_detector.py
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

---

### 2. 꽃 탐지 API 테스트

```bash
curl -X POST "http://127.0.0.1:8000/ai/detect-flowers" \
  -F "image=@test/test_1.jpeg"
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

`.env` 파일을 사용하는 경우:

```bash
docker run --env-file .env -p 8000:8000 bloomary-ai
```

환경변수를 직접 넘기는 경우:

```bash
docker run -p 8000:8000 \
  -e GEMINI_API_KEY=your_actual_gemini_api_key \
  bloomary-ai
```

실행 후 아래 주소에서 API 문서를 확인할 수 있습니다.

```text
http://127.0.0.1:8000/docs
```

---

### 3. Docker 실행 시 필요한 파일

Docker 이미지 빌드 전에 아래 파일들이 로컬에 있어야 합니다.

```text
models/bouquet_yolo11s_v3_704_best.pt
data/floriography.json
.env
```

모델 파일은 git에 포함하지 않으므로, 배포 환경에서도 직접 추가해야 합니다.

---

## Model

현재 사용 모델:

```text
YOLO11s
input image size: 704
model file: models/bouquet_yolo11s_v3_704_best.pt
```

추론 설정:

```text
imgsz=704
conf=0.25
iou=0.65
agnostic_nms=False
max_det=150
```

같은 꽃 이름이 여러 번 탐지된 경우, 서버 내부에서 대표 객체 하나만 남깁니다.

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

---

## Git Ignore Policy

아래 파일들은 git에 포함하지 않습니다.

```text
.env
models/*.pt
test/*.jpeg
test/*.jpg
test/*.png
test/annotated_result.jpg
__pycache__/
*.pyc
```