# cat-detector

Detect a specific cat or cat breed using PyTorch.

We have two indoor cats that each have their own space, in addition to many shared spaces. Beastie is an orange and white stripped Tabby. Mac is a pure black America Shorthair cat. Beastie has a sneaky habit and likes to go into Mac's space some times, which creates tension between them and has resulted in spraying a few times. To help prevent, I'd like to detect when Beastie is in the garage. However, Mac is allowed. The goal is to notify when a tabby is detected only from security camera.

For now I'm just working on the image detection piece using jpg images.

  - [Setup Virtual Python Environment](#setup-virtual-python-environment)
  - [Activate python-env](#activate-python-env)
  - [Install Dependencies](#install-dependencies)
  - [Running](#running)
  - [gocv](#gocv)
  - [Training Custom Model](#training-custom-model)


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


## gocv

Install gocv and opencv 4.10.0 dependencies:
```
mkdir ~/git/github.com/hybridgroup
cd ~/git/github.com/hybridgroup/
git clone https://github.com/hybridgroup/gocv.git
cd gocv
make install
```


The [opencv github repo](https://github.com/opencv/opencv/tree/master/data/haarcascades) includes haarcascades trained classifiers for detecting objects of particular types:
```
mkdir ~/git/github.com/opencv
cd ~/git/github.com/opencv/
git clone https://github.com/opencv/opencv.git
ls -l ~/git/github.com/opencv/opencv/data/haarcascades/haarcascade_frontalcatface.xml
```


/home/bryant/git/github.com/opencv/opencv/data/haarcascades/haarcascade_frontalcatface.xml

### Training Custom Model

Steps used to train custom model:
 - Download the images.tar.gz data set for cats and associated annotations.tar.gz from https://www.robots.ox.ac.uk/~vgg/data/pets/
 - mkdir -p data/{train,val} && cd data && tar -xzf {images,annotations}.tar.gz
 - Removed the dog images: `rm -f data/images/[a-z]*.jpg`
 - Use cat_detector.go to preprocess the images and detect cat with bounding box: `go run cat_detector.go -i data/images`
 - Move bounding box images into data/train/
 - Add my own set of tabby cat detected images into data/train/
 - Copy some of the original images (no bounding box) across breeds from data/images/ into data/val/
 - Add my own set of tabby cat validation images without bounding box into data/val
 - Run cat_breed_train.py to train custom model


Example training run:
```
(python-env) bryant@debian:~/git/github.com/brybry192/cat-detector$ ./cat_breed_train.py
Class Distribution: Counter({'Persian': 166, 'Bombay': 166, 'Abyssinian': 165, 'Maine_Coon': 164, 'British_Shorthair': 164, 'Sphynx': 164, 'Siamese': 163, 'Russian_Blue': 163, 'Ragdoll': 163, 'Bengal': 162, 'Tabby': 128, 'Unknown': 69})
Class Distribution: Counter({'Unknown': 105, 'Bengal': 38, 'Ragdoll': 37, 'Siamese': 37, 'Russian_Blue': 37, 'Maine_Coon': 36, 'Sphynx': 36, 'British_Shorthair': 36, 'Bombay': 34, 'Persian': 34, 'Abyssinian': 33})
Starting training on cuda:0

Epoch    Duration (s)	Phase    Loss     Accuracy
------------------------------------------------------------
1 / 10   50.28       	train    2.3740   0.3576
1 / 10   14.26       	val      2.1403   0.5637
2 / 10   49.64       	train    1.6666   0.7022
2 / 10   15.63       	val      1.3728   0.7495
3 / 10   50.51       	train    0.8995   0.8204
3 / 10   14.20       	val      0.8473   0.7819
4 / 10   49.45       	train    0.5368   0.8683
4 / 10   14.21       	val      0.7707   0.7970
5 / 10   49.61       	train    0.3939   0.8977
5 / 10   14.27       	val      0.7273   0.8229
6 / 10   49.73       	train    0.2996   0.9232
6 / 10   14.04       	val      0.7145   0.8121
7 / 10   49.92       	train    0.2565   0.9260
7 / 10   14.21       	val      0.7205   0.8186
8 / 10   50.07       	train    0.2011   0.9472
8 / 10   14.18       	val      0.7308   0.8229
9 / 10   50.01       	train    0.1810   0.9537
9 / 10   14.09       	val      0.7055   0.8294
10 / 10  49.79       	train    0.1667   0.9521
10 / 10  14.47       	val      0.7175   0.8272

Training complete in 642.35 seconds
Model saved as models/cat_breed_resnet50.pth

```


Using ./cat_breed_detector.py on some test images to see the results:
```
(python-env) bryant@debian:~/git/github.com/brybry192/cat-detector$ ./cat_breed_detector.py images/
Cat not detected in images/english_setter_31.jpg with probability 0.27
Tabby detected in images/IMG_0359.jpg with probability 0.98
Cat not detected in images/IMG_0738.jpg with probability 0.13
Cat not detected in images/IMG_0699.jpg with probability 0.13
Cat not detected in images/IMG_0704.jpg with probability 0.11
Cat not detected in images/scottish_terrier_183.jpg with probability 0.55
Tabby detected in images/IMG_0359_beastie.jpg with probability 0.98
Cat not detected in images/keeshond_134.jpg with probability 0.37
Tabby detected in images/IMG_0366_beastie-cropped-0.jpg with probability 0.70
Cat not detected in images/newfoundland_1.jpg with probability 0.47
Cat not detected in images/yorkshire_terrier_14.jpg with probability 0.26
Cat not detected in images/pug_107.jpg with probability 0.25
Tabby detected in images/IMG_0718.jpg with probability 0.87
Cat not detected in images/english_cocker_spaniel_18.jpg with probability 0.24
Cat not detected in images/great_pyrenees_143.jpg with probability 0.20
Tabby detected in images/IMG_0359_beastie-cropped-0.jpg with probability 0.90
Tabby detected in images/IMG_0366_beastie.jpg with probability 1.00
Cat not detected in images/IMG_3447.jpg with probability 0.30
Cat not detected in images/wheaten_terrier_111.jpg with probability 0.35
Cat not detected in images/shiba_inu_186.jpg with probability 0.46
```




