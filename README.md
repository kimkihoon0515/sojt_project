# SOJT 프로젝트 

## 실행방법

1. backend 폴더로 이동
    ```
    cd backend
    ```
2. 필요한 라이브러리 설치
    - conda
        ```
        conda env create -f conda_requirements.txt
        ```
    - pip
        ```
        pip install -r requirements.txt
        ```
2. MLflow 서버 구동
    ```
    sh track.sh
    ```
3. 학습 진행
    ```
    sh train.sh
    ```

## 학습진행 시 주의할점?
mlflow_practice.py 를 실행할 때 args 들을 다양하게 조건에 맞춰서 주어서 실험하시면 됩니다.  

