# SKN06-2nd-6Team

## 아무래도 귀엽조
2차 단위 프로젝트-정유진, 김승현, 백하은, 박서윤

팀원 소개

| 팀장 | 팀원1 | 팀원2 | 팀원3 |
| --- | --- | --- | --- |
|<img src="https://github.com/user-attachments/assets/5a96b969-a5b1-4d00-8b41-dcc3c51cb747" alt="busil gamza" width="250"/>|<img src="https://github.com/user-attachments/assets/5561d8f1-e92c-400b-8419-44ea0e90466c" alt="seunghyun" width="250"/>|<img src="https://github.com/user-attachments/assets/7b0758fc-6927-474c-a39a-ff72f9360fdf" alt="haeun" width="250"/>|<img src="https://github.com/user-attachments/assets/1b667da5-59ff-405a-aff9-8145e919272a" alt="seoyoon" width="250"/>|
| 정유진 | 김승현 | 백하은 | 박서윤 |


## 1. 프로젝트 소개
   ### 프로젝트 기간
   - 2024.11.13-2024.11.14
 
   ### 개요
   - 모바일 내비게이션 어플인 "Waze"의 데이터 셋을 사용하여, Wasze 어플리케이션 서비스의 잠재 이탈 고객을 파악을 목표로 한다.
     
  ### 배경
   -   모바일 내비게이션 어플의 북미 시장은 "Apple Maps", "Google Maps", "Waze" 3사의 경쟁적 구도를 띄고 있다.
   -   그 중 "Waze"는 사용자 참여형 내비게이션 앱으로 고객 이탈률 파악이 서비스 안정성과 성장에 있어 중요한 요소로 작용한다.
   -   서비스 사용 이탈에 영향을 미치는 특성을 파악하여 향후 서비스 향상을 위한 전략을 세우는 데 활용하고자 한다.
  ### 목적
   - 이탈 가능성이 높은 고객을 파악하고 이를 바탕으로 고객 유지율 향상을 위한 전략을 세울 수 있다.
______________________________________________________________________________________________________


 ## 2. 데이터
  - Waze 데이터
  - Google Advanced Data Analytics Professional Certificate 프로그램의 일부로 제공되는 데이터 셋 사용
  - 앱 내 사용자 행동 로그를 시뮬레이션 하도록 설계되어 있음.
 ### - EDA
  
   ##### a. 데이터 구조
   | Column                   | Data Type | Description                                      |
|--------------------------|-----------|--------------------------------------------------|
| `label`                  | object    | The churn status label indicating churn (if available). |
| `sessions`               | int64     | Number of sessions within a specified period.    |
| `drives`                 | int64     | Number of drives initiated within a specified period. |
| `total_sessions`         | float64   | Total count of sessions recorded for the user.   |
| `n_days_after_onboarding`| int64     | Number of days since the user onboarded.         |
| `total_navigations_fav1` | int64     | Number of times the user navigated to their first saved location. |
| `total_navigations_fav2` | int64     | Number of times the user navigated to their second saved location. |
| `driven_km_drives`       | float64   | Total kilometers driven in recorded drives.      |
| `duration_minutes_drives`| float64   | Total minutes spent driving in recorded drives.  |
| `activity_days`          | int64     | Number of active days recorded for the user.     |
| `driving_days`           | int64     | Number of days the user was engaged in driving.  |
| `device`                 | object    | Type of device used by the user.                 |

      -  관측치: 14,999 개
      -  변수: 13 개
      - `label` 컬럼을 제외한 모든 컬럼은 결측값 없음. `label`은 700개의 결측값을 가짐.
##### b. Target 분포
<p align="center">
  <img src="https://github.com/user-attachments/assets/f9ad1128-e074-4b04-97a8-ad926e5e7e44" alt="Target Distribution" width="600"/>
</p>

##### c. Feature 분포
<p align="center">
  <img src="https://github.com/user-attachments/assets/5ecec3f2-eb65-4355-b9d2-8fa324c82fe8" alt="Feature Distribution" width="600"/>
</p>

##### d. Feature 별 이탈 분포
<p align="center">
  <img src="https://github.com/user-attachments/assets/1119e570-1a62-43d3-8827-6247b22d3bda" alt="Feature Churn Distribution" width="600"/>
</p>

##### e. 다중공선성(Feature 간 상관관계) 확인
<p align="center">
  <img src="https://github.com/user-attachments/assets/2841617a-8baf-4860-996f-175eec8ed526" alt="Multicollinearity Check" width="600"/>
</p>

   
______________________________________________________________________________________________________

## 3. 모델링
 ### a. Machine Learning
 #### - **사용 모델**
 (1) Logistic Regression
 <br>
 (2) Gradient Boosting Machine
 <br>
 (3) Random Forest
 <br>
 (4) K-Nearest Neighbors
 <br>
 (5) XGBoost
 
 #### - **모델 별 평가지표**
  Metric      | Logistic Regression (LR) | Gradient Boosting Machine (GBM) | Random Forest (RF) | K-Nearest Neighbors (KNN) | XGBoost (XGB) |
|-------------|---------------------------|----------------------------------|---------------------|---------------------------|---------------|
| Accuracy    | 0.828322                  | 0.824476                         | 0.821678            | 0.801748                  | 0.811189      |
| Precision   | 0.836831                  | 0.834962                         | 0.835213            | 0.826389                  | 0.838625      |
| Recall      | 0.983000                  | 0.980450                         | 0.975776            | 0.960901                  | 0.954101      |
| F1 Score    | 0.904045                  | 0.901876                         | 0.900039            | 0.888583                  | 0.892644      |
| ROC-AUC     | 0.758722                  | 0.752880                         | 0.732124            | 0.530733                  | 0.709396      |

 #### - **하이퍼파라미터 조정**
 - roc-auc 기준으로 성능이 상대적으로 좋은 LR과 GBM에 대해 하이퍼파라미터 튜닝 진행
 - 학습 결과 평가지표
   
     Metric      | Logistic Regression (LR) | Gradient Boosting Machine (GBM) |
   |-------------|---------------------------|----------------------------------|
   | Accuracy    | 0.825874                  | 0.826224                         |
   | Precision   | 0.833994                  | 0.834294                         |
   | Recall      | 0.984275                  | 0.984275                         |
   | F1 Score    | 0.902924                  | 0.903100                         |
   | ROC-AUC     | 0.739459                  | 0.744959                         |
   
 #### - **모델 별 Feature Importance**
 - 하이퍼 파라미터 조정으로 성능이 크게 개선되지 않아, feature importnace 확인을 진행하여 중요도가 낮은 feature(device)를 제외시켜 학습 재진행.
 - feature importance plot
   <b>

   <p align="center">
  <img src="https://github.com/user-attachments/assets/ddc3873e-06d5-44b8-b10d-a9747ef1f763" alt="Feature importance" width="600"/>
</p>

 #### - **'device' Feature 제외 평가지표** 
 - 최적 파라미터 찾은 후 학습시킨 최종 결과
 - 유의미한 성능 개선이 없었음.
   
      Metric      | Logistic Regression (LR) | Gradient Boosting Machine (GBM) |
   |-------------|---------------------------|----------------------------------|
   | Accuracy    | 0.826224                  | 0.824825                         |
   | Precision   | 0.834053                  | 0.832615                         |
   | Recall      | 0.984700                  | 0.985125                        |
   | F1 Score    | 0.903138                  | 0.902472                         |
   | ROC-AUC     | 0.745144                  | 0.740471                         |
   
 
 ### b. Deep Learning
 #### - **평가지표**
 ##### a. Confusion Matrix
 <p align="center">
 <img src="https://github.com/user-attachments/assets/6e7d35ad-23e8-48d6-99f5-89643be3effd" width="600"/>
 </p>
 
 ##### b. Model Performance Metrics
 <p align="center">
 <img src="https://github.com/user-attachments/assets/a2235ed9-8b05-441f-b26b-1fe2f4b105f7" alt="model performance metrics" width="600"/>
 </p>

##### c. Training Loss and Accuracy Over Epochs
 <p align="center">
 <img src="https://github.com/user-attachments/assets/22b9b199-c7e7-4dae-b1d8-2d1d88ff8b12" width="600"/>
 </p>
 
 ##### d. Receiver Operating Characteristic(ROC) Curve
 <p align="center">
 <img src="https://github.com/user-attachments/assets/8af3eb43-e1fc-4172-bade-9e56057050ff" alt="roc" width="600"/>
 </p>
 
## 4. 결과

## 5. 서비스구현 ( Streamlit )
### 페이지 구성
- About
- Data
- Predictor
- Dashboard

## 6. 회고
정유진:
김승현: 벌써 두번째 프로젝트를 진행하며, 다시 한번 제가 가진 부족함을 깨달았습니다. 그래도 딥러닝 및 머신러닝이 어떤 방식으로 적용되는지 느끼는 계기가 된 것 같습니다.
백하은: 에러를 해결하고 결과를 보았을 때의 쾌감...이래서 코딩하나봅니다.
박서윤:

