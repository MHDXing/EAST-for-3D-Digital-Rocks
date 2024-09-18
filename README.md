# EAST-for-3D-Digital-Rocks
This repository is an official PyTorch implementation of the paper "Efficiently Reconstructing High-Quality Details of 3D Digital Rocks with Super-Resolution Transformer ".

The source code is primarily derived from [EDSR](https://github.com/sanghyun-son/EDSR-PyTorch).
We provide full training and testing codes. You can train your model from scratch, or use a pre-trained model to enlarge your digital rock images.
We will upload the pre-trained model soon.

## Code
### Dependencies
* Python 3.8.5
* PyTorch = 2.0.1
* numpy
* cv2
* skimage
* tqdm


### Quick Start

```bash
git clone ### Quick Start

```bash
git clone https://github.com/MHDXing/MASR-for-Digital-Rock-Images.git
cd EAST-for-3D-Digital-Rocks-main/src
```

## Dataset
The dataset we used was derived from [DeepRockSR-3D](https://www.digitalrocksportal.org/projects/215).
There are 2400, 300, 300 HR 3D images (100x100x100) for training, testing and validation, respectively.

#### Training
1. Download the dataset and unpack them to any place you want. Then, change the ```dir_data``` argument in ```./options.py``` or  ```demo.sh``` to the place where images are located
2. You can change the hyperparameters of different models by modifying the files in the ```./options.py``` 
3. Run ```main.py``` using script file ```demo.sh```
```bash
bash demo.sh
```
4. You can find the results in ```./experiments/EAST``` if the ```save``` argument in ```./options``` is ```EAST```.

#### Testing
1. Download our pre-trained models to ```./models``` folder or use your pre-trained models
2. Change the ```dir_data``` argument in ```./options.py``` or  ```demo.sh``` to the place where images are located
3. Run ```main.py``` using script file ```demo.sh```
```bash
bash demo.sh
```
4. You can find the enlarged images in ```./experiments/results``` folder.
