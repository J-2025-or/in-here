---
title: (First Project)Bridging Data to Deep Learning From Pandas DataFrame to TensorFlow Tensor
layout: post
# post-image: https://raw.githubusercontent.com/thedevslot/WhatATheme/master/assets/images/How%20to%20install%20and%20use%20WhatATheme.png?token=AHMQUEPHRKQFL5FS624RDJ26Z64RDJ26Z64HK
description: 파이썬을 배우고 Pandas DataFrame으로 정리한 데이터를 딥러닝 모델이 효율적으로 '학습'할 수 있도록 TensorFlow Tensor 형태로 전환하는 필수 과정을 작성해봤습니다. 
date: 2025-11-26 # 이 줄을 추가하여 목록 페이지 날짜를 현재로 설정합니다.
tags:
- deep learning for beginners
- python
- pandas
- tensorflow
---
# 🌉 데이터 준비의 첫걸음: Pandas DataFrame을 TensorFlow Tensor로

많은 빅데이터를 인공지능 모델에 최대한 정확하게 집어넣기 제가 공부한 내용을 정리해봈습니다. 

수업에서 파이썬으로 데이터를 분석할 때 사용하는 **Pandas DataFrame** 형태의 데이터를 모델이 이해할 수 있는 **TensorFlow Tensor** 형태로 바꾸는 '다리 놓기(Bridging)' 과정을 작성하고자 합니다. 

---

**TensorFlow**는 데이터를 텐서(Tensor)라는 특별한 구조로 처리해야 합니다. 이 과정은 딥러닝 모델이 데이터를 효율적으로 '학습'하는 데 결정적인 역할을 합니다.

### 1-1. DataFrame과 Tensor의 차이
* **Pandas DataFrame:** 행과 열에 이름이 붙어있고, 문자열, 날짜 등 다양한 데이터 타입을 포함합니다. (사람에게 친숙)
* **TensorFlow Tensor:** 오직 숫자(float, int)만 들어있는 다차원 배열입니다. (컴퓨터/모델에 친숙)
* **목표:** Pandas에서 데이터를 정제하여 모델이 학습할 수 있는 **순수한 숫자 텐서**로 변환하는 것입니다.

### 2. 🧹 Pandas를 이용한 데이터 정제 
데이터 브릿지 작업의 90%는 여기서 끝납니다. 모델에 넣기 전에 데이터를 깨끗하게 만들어야 합니다. 아래의 단계를 반드시 거쳐야 합니다.

* **결측치 및 문자열 처리:** Pandas를 사용하여 데이터에 있는 **결측치**(`NaN`)를 제거하거나 평균값으로 채우고, 문자열은 **One-Hot Encoding** 등을 사용해 숫자로 변환합니다.
* **특성 스케일링 (Scaling):** 데이터의 크기가 클 때 모델이 불안정해질 수 있습니다. 모든 숫자 데이터를 **0과 1 사이** 또는 **평균 0, 분산 1**이 되도록 조정하는 스케일링 작업이 필수적입니다.

### 3. 🌉 NumPy를 거쳐 TensorFlow Tensor로 최종 변환
데이터 정제가 끝났다면, 이제 DataFrame을 NumPy 배열로 변환하는 '중간 다리'를 거쳐 TensorFlow Tensor로 최종 변환합니다. 이 과정이 모델에 데이터를 넣는 마지막 단계입니다.

* **NumPy 배열로 변환:** DataFrame을 `.values` 속성을 이용해 NumPy 배열로 쉽게 변환합니다. 이는 딥러닝 모델이 이해할 수 있는 순수한 숫자 형태입니다.
* **TensorFlow Tensor로 변환:** `tf.convert_to_tensor` 함수를 사용하여 NumPy 배열을 `tf.Tensor` 객체로 변환합니다. 이때 데이터 타입은 보통 **`tf.float32`**로 지정합니다.
* **핵심 코드:** 아래 예시를 참고하여 특성 데이터(X)와 레이블(y)을 텐서로 만들어 보세요.

### 4. 💡 전체 브리징 과정 요약
성공적인 딥러닝 학습은 데이터 준비에 달려 있습니다. 이 3단계를 통해 데이터 브릿지를 완성하고, 이제 여러분이 배운 파이썬 지식을 활용하여 딥러닝 모델의 `fit()` 함수에 텐서를 바로 넣어 학습을 시작할 수 있습니다!

* **단계 1:** Pandas (데이터 정제 및 구조화)
* **단계 2:** NumPy (중간 숫자 배열 형태)
* **단계 3:** TensorFlow Tensor (모델 학습에 최적화된 최종 형태)

#### 📝 참고 코드 (Pandas & TensorFlow)
```python
import pandas as pd
import tensorflow as tf
import numpy as np

# 1. 데이터 불러오기 (가정)
df = pd.read_csv('your_data.csv')

# 2. 정제 (예: 결측치 0 채우기, target 분리)
df = df.fillna(0)
X_data = df.drop('target', axis=1).values 
y_label = df['target'].values

# 3. 텐서로 최종 변환
X_tensor = tf.convert_to_tensor(X_data, dtype=tf.float32)
y_tensor = tf.convert_to_tensor(y_label, dtype=tf.float32)

print(f"X 텐서 형태: {X_tensor.shape}")
print(f"Y 텐서 타입: {y_tensor.dtype}")
# 이제 이 텐서를 모델.fit()에 사용합니다!