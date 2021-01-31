git clone https://github.com/moonitdev/MAI.git


기계학습 

## 환경 설정
- [필요 프레임워크와 라이브러리(딥 러닝을 이용한 자연어 처리 입문)](https://wikidocs.net/25280)

### 가상환경 설정 / 기본 라이브러리 설치
conda create -n mai numpy pandas matplotlib jupyter scikit-learn seaborn nltk

### 라이브러리 설치
#### Tensorflow
- [윈도우 Tensorflow-GPU(2.2.0) 설치하기 - 처음부터 끝까지 상세하게](https://chancoding.tistory.com/89)
- [텐서플로우(tensorflow) 윈도우 10 GPU 설치](https://teddylee777.github.io/colab/tensorflow-gpu-install-windows)

##### tensorflow 기본 라이브러리 설치
```
conda activate mai

pip install tensorflow
pip install tensorflow-gpu
# pip install tf-nightly

python
>>>import tensorflow as tf
>>>tf.__version__
'2.4.1'
```

##### CUDA 설치
- GPU 사양 확인
    - 장치 관리자 > 디스플레이 어댑터
    - Lenovo 노트북: NVDIA GeForce GTX 1060 with Max-Q Design 6G
- Compute Capability 확인
    - [https://developer.nvidia.com/cuda-gpus#compute](https://developer.nvidia.com/cuda-gpus#compute)
    - CUDA-Enabled GeForce and TITAN Products
    - GeForce GTX 1060 : 6.1
- NVIDIA 드라이버 설치
    - [NVIDIA 드라이버](https://www.nvidia.com/Download/index.aspx?lang=kr)
    - GeForce / GeForce 10 Series(Notebooks) / GeForce GTX 1060 / Windows 10 64-bit / Game Ready 드라이버 / Korean
    - 검색 > 다운로드 > 다운로드한 드라이버 설치 > 드라이버 설치 완료
    - cmd: nvidia-smi
- CUDA Toolkit 설치
    - [tested_build_configurations](https://www.tensorflow.org/install/source_windows#tested_build_configurations)
        - tensorflow_gpu-2.4.0	3.6-3.8	MSVC 2019	Bazel 3.1.0	8.0	11.0
    - [cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)
        - CUDA Toolkit 11.2.0 (Dec 2020) > Windows / x86_64 / 10 / exe(network)

##### Visual Studio 설치
- https://docs.microsoft.com/ko-kr/visualstudio/productinfo/2017-redistribution-vs#vs2017-download
- https://docs.microsoft.com/en-us/visualstudio/releases/2019/release-notes
- Download Community 2019
- Visual Studio Installer 실행
- 기본 설치

##### CuDNN 설치
- [cudnn](https://developer.nvidia.com/cudnn)
- 회원 가입
- Download cuDNN v8.1.0 (January 26th, 2021), for CUDA 11.0,11.1 and 11.2 > cuDNN Library for Windows (x86)
- 압축파일 > 압축 해제
- 압축 해제 파일 덮어쓰기 > C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2

##### 환경 변수 설정
- 시스템 환경 변수 편집
- 환경 변수 > 시스템 변수 > path 추가
    - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
    - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\extras\CUPTI\libx64
    - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include

##### 설정 확인
- tensorflow,  keras 버전 확인
```python
import pandas as pd
import tensorflow as tf
tf.__version__

from tensorflow import keras
keras.__version__
```

- GPU 연동 확인
```python
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
```

### Jupyter-가상환경 Kernel 연결
- [Jupyter Notebook에 가상환경 Kernel 연결하기](https://chancoding.tistory.com/86)

#### 미니콘다(아나콘다) 가상 환경 만들기

#### 가상환경 활성화

#### 가상환경에 jupyter notebook 설치

#### 가상환경에 ipykernel 설치
```
pip install ipykernel
```

#### 가상환경에 kernel 연결
```
(mai) >python -m ipykernel install --user --name mai --display-name MAI

python -m ipykernel install --user --name 가상머신이름 --display-name "표시할이름"

```


BUG!
```
2021-01-30 20:20:35.147923: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2021-01-30 20:20:35.156174: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
```

SOL!
- [Could not load dynamic library 'cudart64_101.dll' on tensorflow CPU-only installation](https://stackoverflow.com/questions/59823283/could-not-load-dynamic-library-cudart64-101-dll-on-tensorflow-cpu-only-install)

1. install tf-nightly-gpu
```bash
(mai) > pip install tf-nightly-gpu
```
- 에러 해결 안됨!!!

2. cudart64_101.dll 설치
- [download cudart64_101.dll](https://www.dll-files.com/cudart64_101.dll.html)
- 파일 붙어넣기 > C:\ProgramData\Miniconda3\envs\mai\Library\bin

2021-01-30 20:36:34.810969: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2021-01-30 20:36:34.819272: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.

Could not load dynamic library 'cublas64_11.dll'; dlerror: cublas64_11.dll not found
Could not load dynamic library 'curand64_10.dll'; dlerror: curand64_10.dll not found
Could not load dynamic library 'cudnn64_8.dll'; dlerror: cudnn64_8.dll not found


## 재설치
https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/






[텐서플로우(tensorflow) 윈도우 10 GPU 설치](https://teddylee777.github.io/colab/tensorflow-gpu-install-windows)


[Windows 10에 CUDA Toolkit 11.0 cuDNN 8 Tensorflow 설치하는 방법](https://webnautes.tistory.com/1454)


[병렬처리 위한 GPU, CUDA, cuDNN 삽질기](https://blog.naver.com/PostView.nhn?blogId=euue717&logNo=222098575822)



### BUG!
Could not load dynamic library 'cusolver64_10.dll'; dlerror: cusolver64_10.dll not found

### SOL!
위치: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
파일이름 변경: cusolver64_11.dll > cusolver64_10.dll



