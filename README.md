### 설치 전 준비
```
sudo apt update
sudo apt install software-properties-common

sudo add-apt-repository ppa:deadsnakes/ppa

sudo apt update
sudo apt install python3.7

sudo apt install python3-pip
python3.7 -m pip install pip

```

### 원격 환경 세팅
```
jupyter notebook --generate-config
vi ~/.jupyter/jupyter_notebook_config.py

c.NotebookApp.ip = ~
c.NotebookApp.open_browser = Fasle
```

### 새로운 환경에서 설치할 때

가상환경을 만들고 필요 패키지를 설치해야함

가상환경 패키지인 virtualenv 설치 및 생성(venv)
```
python3.7 -m pip install virtualenv
virtualenv venv --python=python3.7
```

가상환경 실행 및 필요 패키지 설치
```
source venv/bin/activate
pip install -r requirements.txt
```

가상환경 실행
```
nohup jupyter-notebook --ip=0.0.0.0 --port=8888 --allow-root &
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

- [Ubunt에 파이썬 설치](https://softwaree.tistory.com/85)
- [pip 설치](https://stackoverflow.com/questions/54633657/how-to-install-pip-for-python-3-7-on-ubuntu-18)
- [[Keras Study] Mac에서 Keras 환경 구축하기](https://subinium.github.io/Keras-enviroment/)
- [맥에서 케라스 설치하기](https://tykimos.github.io/2017/08/07/Keras_Install_on_Mac/)
