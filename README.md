### 새로운 환경에서 설치할 때

가상환경을 만들고 필요 패키지를 설치해야함

```
    pip install virtualenv // 가상환경 툴 설치
    virtualenv venv // 가상환경 venv 생성
    source venv/bin/activate // 가상환경 venv 실행
    pip install -r requirements.txt // 필요 패키지 설치
```

### 패키지가 변경되었을 때

다른 곳에서도 똑같은 환경을 유지할 수 있도록 필요 패키지 목록 최신화

```
    pip freeze > requirements.txt
```

### 가상환경 종료

```
    deactivate
```

---
### 설치 시 참고 페이지

- [[Keras Study] Mac에서 Keras 환경 구축하기](https://subinium.github.io/Keras-enviroment/)
- [맥에서 케라스 설치하기](https://tykimos.github.io/2017/08/07/Keras_Install_on_Mac/)