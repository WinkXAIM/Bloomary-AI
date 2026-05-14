# Bloomary-AI

## Directory Structure
- **data/**: 프로젝트에 필요한 데이터 보관
- **models/**: 학습된 YOLO 모델 가중치 보관
   - `best.pt`는 용량 및 관리 문제로 git에 포함하지 않습니다.
   - 로컬 실행 시 `models/best.pt` 위치에 직접 추가해주세요.
- **src/server/**: FastAPI 서버 및 API 관련 소스 코드
- **src/logic/**: LLM 프롬프트 및 추천 로직 코드
- **notebooks/**: 학습 실험 및 데이터 분석 기록 (.ipynb)
- **Dockerfile**: 서버 배포를 위한 도커 설정 파일
- **test/**: 로컬 테스트용 코드 및 이미지
   - 테스트 이미지는 git에 포함하지 않습니다.
- **requirements.txt**: 로컬 실행에 필요한 Python 패키지 목록
- **.env.example**: 환경변수 예시 파일

## Branch Strategy
1. **main**: 최종 배포 브랜치 
2. **feature/이슈번호-기능명**: 각자 맡은 기능을 개발하는 작업 공간
   - 예: `feature/1-yolov8-fastapi`, `feature/2-llm-prompt`
   - 작업 완료 시 `main` 브랜치로 Pull Request(PR)를 날려주세요.

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

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. 환경변수 설정

`.env.example` 파일을 복사해서 `.env` 파일을 생성합니다.

```bash
cp .env.example .env
```

Windows PowerShell에서는 아래 명령어를 사용할 수 있습니다.

```bash
copy .env.example .env
```

`.env` 파일에 Gemini API Key를 입력합니다.

```env
GEMINI_API_KEY=your_actual_gemini_api_key
```

### 4. YOLO 모델 파일 추가

YOLO 모델 가중치 파일은 git에 포함하지 않습니다.

아래 경로에 모델 파일을 직접 추가해주세요.

```text
models/best.pt
```

### 5. 로컬 테스트 이미지 추가

테스트 이미지를 아래 경로에 추가합니다.

```text
test/test_bouquet.jpeg
```

테스트 이미지는 git에 커밋하지 않습니다.

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
      "meaning": "",
      "box2d": [100, 200, 300, 400],
      "color_hex": "#D94B5F"
    }
  ]
}
```
