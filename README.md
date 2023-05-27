# A General Keyless Extraction Framework Targeting at Deep Learning Based Image-within-image Models
 
## Dependencies and Installation
- Python 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux)).
- [PyTorch = 1.7.1](https://pytorch.org/) .
- See requirement.txt for other dependencies.


## Get Started
- Run `python train.py` for training.

- Run `python extraction.py` for testing.

- Download link (Google Drive) for our trained model weights: https://drive.google.com/file/d/1e5gozAXTzVaFW8sOLeML33KhbOz6FUIj/view?usp=sharing

## Dataset
- In this paper, we use the commonly used dataset ImageNet, COCO and DIV2K.

- You can store your own datasets in the format of our directory 'datasets':

  - `covers`
  - `secrets` 
  - `containers` 
  - `revealedSecrets`

- In train.py or extract.py, find the ImageHidingDataset, change the 'root' parameter, and the code will automatically read the data.

## Results
-  The visualized results are in the 'results' directory
