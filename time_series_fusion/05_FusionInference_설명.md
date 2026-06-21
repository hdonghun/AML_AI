# 05_FusionInference 설명

## 개요

노이즈가 추가된 이미지(`noise_pic/`) 1장 단위로 이상 여부를 예측.  
이미지 모델(MobileNetV2)과 센서 모델(RNN)의 이상 확률을 가중 합산해 최종 판정.

```
final_prob = W_IMG × p_img + W_SENSOR × p_sensor   (W_IMG + W_SENSOR = 1.0)
```

---

## 입력 데이터

| 항목 | 경로 | 형식 |
|---|---|---|
| 노이즈 이미지 | `noise_pic/` | `Val1_f000090.png` 등 Val* 파일명 |
| 센서 CSV | `original_INC_202501/Val*/` | Avg Voltage, Avg Current |
| 라벨 | `original_INC_202501/label_mapping.txt` | trial별 0=정상, 1=이상 |
| 이미지 모델 | `noise_target/best_mobilenet.pth` | MobileNetV2 fine-tuned |
| 센서 모델 | `noise_target/best_rnn.pt` | 2-layer RNN |

### 이미지 파일명 규칙

```
Val1_f000090.png
Val1_130A_22TS_170WFR_f000090.png
Val1_f000090_noise.png
```

- `Val1` 부분 → trial prefix (대소문자 무관)
- `_f000090` 부분 → 원본 영상 프레임 번호 (frame_idx)

---

## 센서 윈도우 정렬 방식

이미지의 frame_idx를 센서 샘플 인덱스로 변환 후, 해당 시점을 **끝**으로 하는 30샘플 윈도우 사용.

```
sensor_idx = round(frame_idx / VIDEO_FPS * SENSOR_HZ)
window     = sensor_data[sensor_idx-29 : sensor_idx+1]   # shape (30, 4)
```

| 파라미터 | 값 | 의미 |
|---|---|---|
| `VIDEO_FPS` | 30 | 원본 영상 FPS |
| `SENSOR_HZ` | 10 | 센서 샘플링 주파수 |
| `WINDOW_SIZE` | 30 | 윈도우 크기 (3초) |

**스킵 조건**: `sensor_idx < 29` (영상 시작 후 약 2.9초 이내 프레임)  
→ 윈도우 앞쪽 샘플이 부족한 이미지는 패딩 없이 제외.

---

## 모델 출력

| 모델 | 입력 | 출력 |
|---|---|---|
| MobileNetV2 | 이미지 1장 (224×224) | `softmax[:, 1]` → p_abnormal (0~1) |
| RNN | 센서 윈도우 (30, 4) — V, I, TS, WFR | `sigmoid` → p_abnormal (0~1) |

### RNN 아키텍처 (best_rnn.pt 저장 시 파라미터)

```
RNN(4 → 32) → Dropout(0.5) → RNN(32 → 16) → Dropout(0.5) → Linear(16 → 1) → Sigmoid
```

### 센서 전처리 (04와 동일)

1. trial 내부 z-score 정규화 (V, I)
2. TS, WFR는 폴더명에서 파싱 후 상수로 붙임
3. 글로벌 StandardScaler (Train split 기준 fit)

---

## 가중치 설정

### 현재: 고정 가중치

```python
W_IMG    = 0.5
W_SENSOR = 0.5
```

### 오토인코더 연동 후: 동적 가중치

`fu_009` 셀에서 `get_weights()` 호출 시 `noise_level` 인자를 전달.

```python
# 오토인코더 출력값(재구성 오차)을 0~1로 스케일링 후 전달
w_i, w_s = get_weights(noise_level=ae_output)
```

`get_weights()` 내부 공식 (`fu_004` 셀):

```python
w_img = clip(1.0 - noise_level, 0.0, 1.0)
# 노이즈가 클수록 W_IMG 감소 → 센서 모델을 더 신뢰
```

---

## 셀 구성

| 셀 ID | 내용 |
|---|---|
| `fu_002` | 경로 / 가중치 / 센서·이미지 설정 |
| `fu_003` | 라이브러리 임포트 |
| `fu_004` | `get_weights()` — 오토인코더 연동 포인트 |
| `fu_005` | MobileNetV2 + RNN 로드, 단일 추론 함수 정의 |
| `fu_006` | `label_mapping.txt` 파싱 |
| `fu_007` | 센서 CSV 로드 + 글로벌 스케일러 fit |
| `fu_008` | `get_sensor_window()` + `parse_image_info()` |
| `fu_009` | 이미지 단위 융합 추론 메인 루프 |
| `fu_010` | 평가 (Accuracy, F1, ROC-AUC, trial별 집계) |
| `fu_011` | 시각화 (확률 시계열, 분포, 혼동행렬) |
| `fu_012` | 가중치 민감도 분석 (W_IMG 0→1 스윕) |

---

## 평가 출력

- **Image-level**: 이미지 1장 단위 Accuracy / Precision / Recall / F1 / ROC-AUC / PR-AUC
- **Trial별 집계**: trial 단위 평균 확률 및 정확도
- **시각화**: 확률 시계열(trial별 색상) + Score 분포 히스토그램 + 혼동행렬
- **민감도 분석**: W_IMG를 0.05 단위로 변화시키며 최적 가중치 탐색

---

## 실행 전 체크리스트

- [ ] `noise_pic/` 폴더에 `Val*_f숫자.png` 형식 이미지 추가
- [ ] `best_mobilenet.pth` / `best_rnn.pt` 경로 확인
- [ ] `VIDEO_FPS` 값이 실제 촬영 FPS와 일치하는지 확인
- [ ] 오토인코더 추가 후 `fu_004` 공식 및 `fu_009` 호출부 수정
