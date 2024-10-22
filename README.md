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

To active the python-env:
```
source python-env/bin/activate
```

## Install dependencies with pip

```
pip install opencv-python numpy==1.26.4 torch torchvision
```


## Running

```
./cat_detector.py images/IMG_0359_beastie.jpg
```



