## 목차

* [1. 시스템 환경](#1-시스템-환경)
  * [1-1. 주요 Python 라이브러리](#1-1-주요-python-라이브러리)
  * [1-2. 시스템에 설치된 전체 Python 라이브러리](#1-2-시스템에-설치된-전체-python-라이브러리)
* [2. 사용자 가이드](#2-사용자-가이드)
  * [2-1. Python 환경 설정](#2-1-python-환경-설정)
  * [2-2. 모델 다운로드 및 준비](#2-2-모델-다운로드-및-준비)
  * [2-3. 기본 실행 방법](#2-3-기본-실행-방법)

## 1. 시스템 환경

* OS & GPU
  * OS : Windows 10
  * GPU : 2 x Quadro M6000 (12 GB each)
* CUDA
  * CUDA 12.4 (NVIDIA-SMI 551.61)
  * ```nvcc -V``` 명령어 실행 시 다음과 같이 표시

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Thu_Jun_11_22:26:48_Pacific_Daylight_Time_2020
Cuda compilation tools, release 11.0, V11.0.194
Build cuda_11.0_bu.relgpu_drvr445TC445_37.28540450_0
```

* Python
  * Python : Python 3.10.11
  * Dev Tool : PyCharm 2024.1 Community Edition

### 1-1. 주요 Python 라이브러리

```
bitsandbytes==0.45.3
huggingface_hub==0.36.2
matplotlib==3.7.5
numpy==1.26.4
opencv-python==4.6.0.66
opencv-python-headless==4.11.0.86
optuna==4.4.0
pandas==2.0.3
peft==0.13.2
pillow==10.2.0
protobuf==3.19.6
safetensors==0.4.3
scikit-image==0.21.0
scikit-learn==1.3.2
scikit-learn-intelex==2025.5.0
scipy==1.10.1
sentence-transformers==4.1.0
sentencepiece==0.2.0
tab-transformer-pytorch==0.6.1
timm==1.0.15
tokenizers==0.21.1
torch==2.6.0+cu124
torchaudio==2.6.0+cu124
torchinfo==1.8.0
torchmetrics==1.7.1
torchvision==0.21.0+cu124
tqdm==4.67.1
transformers==4.51.3
trl==0.19.1
```

### 1-2. 시스템에 설치된 전체 Python 라이브러리

* 본 프로젝트 개발 환경에 설치된 전체 Python 라이브러리의 목록입니다.

<details><summary>전체 Python 라이브러리 목록 [ 펼치기 / 접기 ]</summary>

```
absl-py==2.1.0
accelerate==1.12.0
aiohappyeyeballs==2.4.4
aiohttp==3.10.11
aiosignal==1.3.1
albumentations==1.3.1
alembic==1.16.4
annotated-doc==0.0.4
annotated-types==0.7.0
antlr4-python3-runtime==4.9.3
anyio==4.12.1
asgiref==3.11.0
astunparse==1.6.3
async-timeout==4.0.3
attrs==25.3.0
auto_gptq==0.7.1
backoff==2.2.1
bcrypt==5.0.0
beartype==0.22.9
beautifulsoup4==4.13.3
bitsandbytes==0.45.3
bs4==0.0.2
build==1.4.0
cachetools==5.3.3
certifi==2024.2.2
charset-normalizer==3.3.2
chroma-hnswlib==0.7.3
chromadb==0.5.0
click==8.3.1
colorama==0.4.6
coloredlogs==15.0.1
colorlog==6.9.0
contourpy==1.1.1
cut-cross-entropy==25.1.1
cycler==0.12.1
Cython==3.0.12
daal==2025.5.0
datasets==3.5.0
Deprecated==1.2.18
diffusers==0.32.2
dill==0.3.8
discrete-continuous-embed-readout==0.2.2
distro==1.9.0
docstring_parser==0.16
durationpy==0.10
einops==0.8.2
et_xmlfile==2.0.0
eval_type_backport==0.2.2
exceptiongroup==1.3.1
fastapi==0.128.0
filelock==3.13.4
flatbuffers==1.12
fonttools==4.51.0
frozenlist==1.5.0
fsspec==2024.3.1
gast==0.4.0
gekko==1.2.1
google-auth==2.28.1
google-auth-oauthlib==0.4.6
google-pasta==0.2.0
googleapis-common-protos==1.72.0
graphviz==0.20.3
greenlet==3.2.3
grpcio==1.76.0
h11==0.16.0
h5py==3.10.0
hf_transfer==0.1.9
httpcore==1.0.9
httptools==0.7.1
httpx==0.28.1
huggingface_hub==0.36.2
humanfriendly==10.0
hydra-core==1.3.2
hyper-connections==0.4.9
idna==3.6
imageio==2.35.1
imgaug==0.4.0
importlib-metadata==7.0.1
importlib_resources==6.4.0
intel-extension-for-transformers==1.4.2
Jinja2==3.1.3
jiter==0.12.0
joblib==1.4.2
jsonpatch==1.33
jsonpointer==3.0.0
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
kaleido==0.2.1
keras==2.8.0
Keras-Preprocessing==1.1.2
kiwisolver==1.4.5
kubernetes==35.0.0
langchain==1.2.10
langchain-classic==1.0.1
langchain-core==1.2.14
langchain-huggingface==1.2.0
langchain-text-splitters==1.1.1
langgraph==1.0.9
langgraph-checkpoint==4.0.0
langgraph-prebuilt==1.0.8
langgraph-sdk==0.3.8
langsmith==0.7.5
lazy_loader==0.4
libclang==16.0.6
lightning==2.1.2
lightning-utilities==0.14.3
Mako==1.3.10
Markdown==3.5.2
markdown-it-py==3.0.0
MarkupSafe==2.1.5
matplotlib==3.7.5
mdurl==0.1.2
mmh3==5.2.0
monai==1.4.0
mpmath==1.3.0
multidict==6.1.0
multiprocess==0.70.16
narwhals==1.33.0
networkx==3.1
neural_compressor==3.3
numpy==1.26.4
oauthlib==3.2.2
omegaconf==2.3.0
onnxruntime==1.23.2
openai==2.16.0
opencv-python==4.6.0.66
opencv-python-headless==4.11.0.86
openpyxl==3.1.5
opentelemetry-api==1.39.1
opentelemetry-exporter-otlp-proto-common==1.39.1
opentelemetry-exporter-otlp-proto-grpc==1.39.1
opentelemetry-instrumentation==0.60b1
opentelemetry-instrumentation-asgi==0.60b1
opentelemetry-instrumentation-fastapi==0.60b1
opentelemetry-proto==1.39.1
opentelemetry-sdk==1.39.1
opentelemetry-semantic-conventions==0.60b1
opentelemetry-util-http==0.60b1
opt-einsum==3.3.0
optimum==1.23.3
optuna==4.4.0
orjson==3.11.6
ormsgpack==1.12.2
overrides==7.7.0
packaging==26.0
pandas==2.0.3
peft==0.13.2
pillow==10.2.0
plotly==6.0.1
posthog==5.4.0
prettytable==3.11.0
propcache==0.2.0
protobuf==3.19.6
psutil==7.0.0
py-cpuinfo==9.0.0
pyarrow==17.0.0
pyasn1==0.5.1
pyasn1-modules==0.3.0
pybase64==1.4.3
pycocotools==2.0.8
pydantic==2.12.5
pydantic_core==2.41.5
pydot==2.0.0
Pygments==2.19.1
pyparsing==3.1.2
PyPika==0.50.0
pyproject_hooks==1.2.0
pyreadline3==3.5.4
python-dateutil==2.9.0.post0
python-dotenv==1.2.1
python-version==0.0.2
pytorch-lightning==2.5.1.post0
pytz==2024.1
PyWavelets==1.4.1
PyYAML==6.0.1
qudida==0.0.4
referencing==0.37.0
regex==2023.12.25
requests==2.32.3
requests-oauthlib==1.3.1
requests-toolbelt==1.0.0
rich==13.9.4
rouge==1.0.1
rpds-py==0.30.0
rsa==4.9
safetensors==0.4.3
schema==0.7.7
scikit-image==0.21.0
scikit-learn==1.3.2
scikit-learn-intelex==2025.5.0
scipy==1.10.1
seaborn==0.13.2
sentence-transformers==4.1.0
sentencepiece==0.2.0
shapely==2.0.7
shellingham==1.5.4
shtab==1.7.1
six==1.16.0
sniffio==1.3.1
soupsieve==2.6
SQLAlchemy==2.0.42
starlette==0.50.0
sympy==1.13.1
tab-transformer-pytorch==0.6.1
tbb==2022.1.0
tcmlib==1.3.0
tenacity==9.1.2
tensorboard==2.8.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
tensorflow-estimator==2.9.0
tensorflow-gpu==2.8.0
tensorflow-io-gcs-filesystem==0.31.0
termcolor==2.4.0
tf-estimator-nightly==2.8.0.dev2021122109
tfutil==0.8.1
threadpoolctl==3.5.0
tifffile==2023.7.10
timm==1.0.15
tokenizers==0.21.1
tomli==2.2.1
torch==2.6.0+cu124
torchaudio==2.6.0+cu124
torchinfo==1.8.0
torchmetrics==1.7.1
torchview==0.2.6
torchvision==0.21.0+cu124
tqdm==4.67.1
transformers==4.51.3
triton-windows==3.2.0.post17
trl==0.19.1
ttach==0.0.3
typeguard==4.4.0
typer==0.21.1
typing-inspection==0.4.2
typing_extensions==4.14.1
tyro==0.9.17
tzdata==2025.2
unsloth==2025.3.19
unsloth_zoo==2025.3.17
urllib3==2.2.1
uuid_utils==0.14.0
uvicorn==0.40.0
watchfiles==1.1.1
wcwidth==0.2.13
websocket-client==1.9.0
websockets==16.0
Werkzeug==3.0.1
wrapt==1.16.0
x-mlps-pytorch==0.3.1
xformers==0.0.29.post3
xxhash==3.5.0
yarl==1.15.2
zipp==3.17.0
zstandard==0.25.0
```

</details>

## 2. 사용자 가이드

### 2-1. Python 환경 설정

* System Requirements
  * **8 GB** 이상 메모리의 **GPU 1 대 (NVIDIA GPU)**
* Mandatory
  * Python 3.10.11 을 설치합니다.
  * ```pip install -r requirements.txt``` 명령어를 통해 [주요 라이브러리](#1-1-주요-python-라이브러리) 를 설치합니다.
* Optional
  * 주요 라이브러리 설치 후에도 Python 라이브러리 이슈로 실행이 안 될 시, [전체 라이브러리 목록](#1-2-시스템에-설치된-전체-python-라이브러리) 을 참고하여 라이브러리를 추가 설치합니다.

### 2-2. 모델 다운로드 및 준비

* **1.** [해당 문서](MODEL_INFO.md) 를 참고하여, HuggingFace 에서 **필요한 모델을 다운로드하여 지정된 경로에 추가** 합니다.
* **2.** [해당 문서](../LLM_DOWNLOAD_PATH.md) 를 참고하여 원본 LLM 모델을 다운로드한 후, **1.** 에서 다운로드한 ```adapter_config.json``` 파일을 변경합니다.

### 2-3. 기본 실행 방법

**1.** ```hpo_run_battle``` 경로에서 ```python run_battle_vs_human.py``` 를 실행합니다.

```
(venv) PS C:\Users\20151\Documents\AI_Projects> cd 2025_10_06_OhLoRA_HP_Battle
(venv) PS C:\Users\20151\Documents\AI_Projects\2025_10_06_OhLoRA_HP_Battle> cd hpo_run_battle
(venv) PS C:\Users\20151\Documents\AI_Projects\2025_10_06_OhLoRA_HP_Battle\hpo_run_battle> python run_battle_vs_human.py
```

**2.** 다음과 같은 문구가 나타나면, [```hpo_run_battle/battle_dataset/hps.txt```](hpo_run_battle/battle_dataset/hps.txt) 파일을 수정한 후 엔터 (Enter) 키를 누릅니다.

* [```hpo_run_battle/battle_dataset```](hpo_run_battle/battle_dataset) 에서 데이터셋을 확인한 후, 하이퍼파라미터 값을 적절히 결정합니다.
  * ```train_dataset``` (학습 데이터셋), ```valid_dataset``` (검증 데이터셋), ```test_dataset``` (테스트 데이터셋) 을 각각 확인합니다. 
* 이것을 각 데이터셋 (CIFAR-10, Fashion-MNIST, MNIST) 에 대해 반복합니다.

**[ 승패 판정 기준 ]**

* 사용자에게 **총 2번의 기회** 를 제공합니다.
* **Oh-LoRA 👱‍♀️ 가 결정한 하이퍼파라미터** 에 비해 최종 점수 (Macro F1 Score) 가 **1번이라도 더 좋으면, 사용자의 승리** 입니다.

```
다음과 같은 형식으로 하이퍼파라미터를 저장하여 battle_dataset/hps.txt 로 저장한 다음 Enter 키를 눌러 주세요.
(이미 battle_dataset/hps.txt 파일이 있다면 최적의 하이퍼파라미터로 수정해 주세요.)

{"dropout_conv_earlier": {0.0 - 0.3 사이의 float 값},
 "dropout_conv_later": {0.0 - 0.3 사이의 float 값},
 "dropout_fc": {0.0 - 0.6 사이의 float 값},
 "lr": {0.00002 - 0.006 사이의 float 값},
 "activation_func": "{relu|leaky_relu}",
 "optimizer": "{adam|adamw}",
 "scheduler": "{exp_80|exp_90|exp_95|exp_98|cosine}"}

(참고: lr은 learning rate 이고, scheduler 중 exp_N 에서 N은 gamma 값 (%) 을 의미합니다.)
```

**3.** 각 데이터셋 별 최종 결과 및 이에 대해 Oh-LoRA 👱‍♀️ LLM이 생성한 문구를 확인합니다.

```
[ 인간의 Macro F1 Score = 0.0674 ]
[ Oh-LoRA 👱‍♀️ 의 Macro F1 Score = 0.4315 ] (예측: 0.3952408730983734)
[ Oh-LoRA 👱‍♀️ 의 하이퍼파라미터 = {'dropout_conv_earlier': 0.0, 'dropout_conv_later': 0.0, 'dropout_fc': 0.6, 'lr': 0.000278296826514608, 'activation_func': 'relu', 'optimizer': 'adam', 'scheduler': 'exp_98'} ]
[ 최종 결과 : 인간 사용자의 승리 ]

 ==== 상세 점수 ====
Human 1st : 0.454
Human 2nd : 0.0674
Oh-LoRA   : 0.4315
=================


Oh-LoRA 👱‍♀️ :  첫 판 덕분에 살았네! 두 번째 판은 망했지만 이겼으니 축하해! ✨
```
