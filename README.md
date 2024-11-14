# SKN06-2nd-6Team
2차 단위 프로젝트-정유진, 김승현, 백하은, 박서윤

팀원 소개

| 팀장 | 팀원1 | 팀원2 | 팀원3 |
| --- | --- | --- | --- |
||||
| --- | --- | --- | --- |

### 1. 프로젝트 소개
   
 
   #### 개요
   - 모바일 내비게이션 어플인 "Waze"의 데이터 셋을 사용하여, Wasze 어플리케이션 서비스의 잠재 이탈 고객을 파악을 목표로 한다.
     
  #### 배경
   -   모바일 내비게이션 어플의 북미 시장은 "Apple Maps", "Google Maps", "Waze" 3사의 경쟁적 구도를 띄고 있다.
   -   그 중 "Waze"는 사용자 참여형 내비게이션 앱으로 고객 이탈률 파악이 서비스 안정성과 성장에 있어 중요한 요소로 작용한다.
   -   서비스 사용 이탈에 영향을 미치는 특성을 파악하여 향후 서비스 향상을 위한 전략을 세우는 데 활용하고자 한다.
  #### 목적
   - 이탈 가능성이 높은 고객을 파악하고 이를 바탕으로 고객 유지율 향상을 위한 전략을 세울 수 있다.

### 2. 데이터
  - Google Advanced Data Analytics Professional Certificate 프로그램의 일부로 제공되는 데이터 셋 사용
  - 앱 내 사용자 행동 로그를 시뮬레이션 하도록 설계되어 있음.
 #### - EDA
  
   ##### (1) 데이터 구조
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
      -  변수: 12 개
      - `label` 컬럼을 제외한 모든 컬럼은 결측값 없음. `label`은 700개의 결측값을 가짐.
   ##### (2) Target 분포
   ![label_distribution](https://github.com/user-attachments/assets/f9ad1128-e074-4b04-97a8-ad926e5e7e44)
   ##### (3) Feature 분포
   ![feautre_plot](https://github.com/user-attachments/assets/5ecec3f2-eb65-4355-b9d2-8fa324c82fe8)
   ##### (4) Feature 별 이탈 분포
   ![target_plot](https://github.com/user-attachments/assets/1119e570-1a62-43d3-8827-6247b22d3bda)
   ##### (5) 다중공선성(Feature 간 상관관계) 확인
   ![corr](https://github.com/user-attachments/assets/2841617a-8baf-4860-996f-175eec8ed526)
   

### 3. 모델링
 #### ML
 #### DL
 
### 4. 결과

### 3. 회고
