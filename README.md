# 🚗 운수종사자 교통사고 위험 예측 AI

[![Competition](https://img.shields.io/badge/Dacon-Competition-blue)](https://dacon.io/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3+-orange)](https://lightgbm.readthedocs.io/)

운수종사자 인지적 특성 데이터를 활용한 교통사고 위험 예측 AI 경진대회 참가 리포지토리

---

## 📋 목차

- [대회 개요](#-대회-개요)
- [프로젝트 구조](#-프로젝트-구조)
- [설치 방법](#-설치-방법)
- [사용법](#-사용법)
- [모델 성능](#-모델-성능)
- [제출 가이드](#-제출-가이드)
- [평가 지표](#-평가-지표)

---

## 🎯 대회 개요

### 대회 정보
- **주최/주관**: 행정안전부, 한국지능정보사회진흥원 / 한국교통안전공단
- **운영**: 데이콘
- **상금**: 총 6,000만원
- **기간**: 2024.10.13 ~ 2024.11.14

### 목표
운수종사자 자격검사 데이터를 활용하여 **교통사고 위험군에 속할 확률(0~1)**을 예측하는 AI 모델 개발

### 데이터 구성
- **Train A**: 신규 자격 검사 (647,241 samples)
  - A1~A5: 행동반응 측정 (반응시간, 정확도)
  - A6~A7: 문제풀이식 검사
  - A8~A9: 질문지형 검사 (심리 척도)

- **Train B**: 자격 유지 검사 (297,526 samples)
  - B1~B8: 반응 기반 검사 (정확도, 반응시간)
  - B9~B10: 신호 탐지 이론 기반 (hit, miss, fa, cr) + 멀티태스킹

- **Test**: 162,216 samples

---

## 🚀 사용법

### 1. 모델 학습 (버전 관리)

```bash
# 기본 학습 (자동 타임스탬프 버전)
python -m src.train --config configs/default.yaml --folds 5

# 커스텀 버전 이름 지정
python -m src.train --config configs/default.yaml --folds 5 --version xgb_baseline_v1
```

**학습 결과 (버전별 폴더에 저장):**
```
model/
├── xgb_20251018_193000/          # 타임스탬프 버전
│   ├── metadata.json              # 학습 메타데이터
│   ├── summary.json               # 전체 성능 요약
│   ├── results_A.json             # Test A 상세 결과
│   ├── results_B.json             # Test B 상세 결과
│   ├── model_A_0.pkl ~ model_A_4.pkl
│   ├── model_B_0.pkl ~ model_B_4.pkl
│   ├── cal_A.pkl, cal_B.pkl
│   ├── feature_cols_A.json
│   └── feature_cols_B.json
├── lgbm_20251018_180000/          # 이전 버전
├── LATEST_VERSION.txt             # 최신 버전 포인터
└── latest/                        # 최신 버전 심볼릭 링크 (선택)
```

### 2. 모델 버전 조회

```bash
# 모든 버전 목록 및 성능 비교
python -m src.utils.version --list

# 최신 버전 확인
python -m src.utils.version --latest

# 두 버전 비교
python -m src.utils.version --compare xgb_20251018_193000 lgbm_20251018_180000
```

### 3. 추론 및 제출 파일 생성

```bash
# 최신 모델로 추론
python -m src.infer --config configs/default.yaml --out submission.csv

# 특정 버전으로 추론 (TODO: 구현 필요)
# python -m src.infer --config configs/default.yaml --version xgb_baseline_v1 --out submission.csv
```

**출력:**
- `submission.csv`: 제출용 예측 결과 파일

### 4. 제출 파일 생성

제출 시 `src/` 코드를 독립 실행 가능한 `script.py`로 변환하여 제출

```bash
# 제출용 스크립트 생성 (수동 또는 자동화 스크립트 사용)
# 1. src/feature_engineering.py의 모든 피처 함수 복사
# 2. src/infer.py의 추론 로직 복사
# 3. 경로를 상대 경로로 변경 (data/, model/, output/)
# 4. src 모듈 import 제거하고 독립 실행 가능하게 수정
```

**제출 파일 구조:**
```
submit.zip
├── script.py              # 독립 실행 추론 스크립트
├── requirements.txt       # 의존성
└── model/                # 모델 파일 (model/ 폴더 전체 복사)
    ├── model_A_*.pkl
    ├── model_B_*.pkl
    ├── cal_A.pkl
    ├── cal_B.pkl
    ├── feature_cols_A.json
    └── feature_cols_B.json
```

**패키징:**
```bash
# 1. model/ 폴더 그대로 사용
# 2. script.py 생성 (src 코드 기반)
# 3. requirements.txt 복사
# 4. 압축
zip -r submit.zip script.py requirements.txt model/
```

---

## 📊 모델 성능

### Cross-Validation 결과 (5-Fold)

#### Test A (신규 자격 검사)
| Metric | Score |
|--------|-------|
| **AUC** | 0.6679 |
| **Brier Score** | 0.02197 |
| **ECE** | 0.00218 |
| **Competition Score** | ~0.17209 |

- **데이터**: 647,241 samples
- **피처**: 73 features
- **양성 비율**: 2.27% (불균형)

#### Test B (자격 유지 검사)
| Metric | Score |
|--------|-------|
| **AUC** | 0.5706 |
| **Brier Score** | 0.04049 |
| **ECE** | 0.00537 |
| **Competition Score** | ~0.22616 |

- **데이터**: 297,526 samples
- **피처**: 56 features
- **양성 비율**: 4.23%

### 주요 피처

**Test A:**
- 반응시간 통계 (mean, std, CV)
- 조건별 반응시간 차이 (좌우, 속도)
- Stroop 효과 지표
- 속도-정확도 트레이드오프

**Test B:**
- 신호 탐지 이론 기반 (d-prime, criterion)
- hit rate, false alarm rate
- 멀티태스킹 성능 (청각 + 시각)
- 시각 과제 오류율

---

## 📤 제출 가이드

### 제출 파일 요구사항

1. **파일 구조** (필수)
   ```
   submit.zip
   ├── model/              # 모델 가중치
   ├── script.py           # 추론 스크립트
   └── requirements.txt    # 패키지 의존성
   ```

2. **평가 서버 환경**
   - OS: Ubuntu 22.04.5 LTS
   - Python: 3.10.12
   - CPU: 3 vCPU / 28GB RAM (GPU 없음)
   - 실행 시간 제한: 30분
   - 패키지 설치 제한: 10분
   - 용량 제한: 10GB
   - 인터넷: ❌ 비활성화

3. **출력 형식**
   - 파일명: `output/submission.csv`
   - 컬럼: `Test_id`, `Label`
   - Label: 교통사고 위험군 확률 (0~1)

### 제출 시 주의사항

- ✅ `requirements.txt`에 필요한 모든 패키지 명시
- ✅ 상대 경로 사용 (`data/`, `model/`, `output/`)
- ✅ 30분 이내 실행 가능하도록 최적화
- ❌ 외부 서버 접속 금지
- ❌ GPU 사용 불가

---

## 📐 평가 지표

### 리더보드 산식

```
Score = 0.5 × (1 - AUC) + 0.25 × Brier + 0.25 × ECE
```

- **낮을수록 좋음** (0점 만점)
- Public Score: 전체 테스트 데이터 100% 기준

### 각 지표 설명

1. **AUC (Area Under ROC Curve)**
   - 분류 성능 측정
   - 범위: 0~1 (1에 가까울수록 좋음)

2. **Brier Score**
   - 확률 예측 정확도
   - 범위: 0~1 (0에 가까울수록 좋음)

3. **ECE (Expected Calibration Error)**
   - 확률 보정 정도
   - 범위: 0~1 (0에 가까울수록 좋음)

---

## 🏆 대회 일정

| 날짜 | 일정 |
|------|------|
| 10.13 | 대회 시작 |
| 11.07 | 팀 병합 마감 |
| 11.13 | 리더보드 제출 마감 |
| 11.14 | 대회 종료 |
| 11.19 | 2차 평가 자료 제출 마감 |
| 11.28 | 최종 결과 발표 |
| 12.03 | 오프라인 시상식 |

---

## 🔧 기술 스택

- **언어**: Python 3.11
- **ML 프레임워크**: LightGBM 4.3+
- **데이터 처리**: pandas 2.2+, numpy 1.26+
- **모델 평가**: scikit-learn 1.5+
- **설정 관리**: PyYAML
- **기타**: joblib (모델 직렬화)

---

## 📝 라이선스

이 프로젝트는 대회 참가 목적으로 작성되었습니다.

---

## 👤 Author

- **대회 참가자**: [Your Name]
- **GitHub**: [Your GitHub]
- **Contact**: [Your Email]

---

## 🙏 Acknowledgments

- 데이콘 운영진
- 행정안전부, 한국지능정보사회진흥원
- 한국교통안전공단
