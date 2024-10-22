# cat-detector
Detect a cat using PyTorch

## Setup Virtual Python Environment

Install dependencies and create python environment for this repo:
```
mkdir models
brew install pyenv
pyenv versions
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
pyenv versions
pyenv global system
python3 -m venv python-env
```

## Activate python-env

Activate the python-env before doing setup or running script::
```
source python-env/bin/activate
```

## Install Dependencies

Install python certificates for os version:
```
cd /Applications/Python\ 3.11/
bash Install\ Certificates.command
```

Install python dependencies with pip:
```
pip install opencv-python numpy==1.26.4 torch torchvision imgbeddings huggingface_hub==0.24 tflite-support psycopg2
```

## Running

```
./cat_detector.py images/IMG_0359_beastie.jpg
```



