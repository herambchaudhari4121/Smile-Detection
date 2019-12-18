# Smile-Detection

Dependency: 

Required libraries and packages can be download from requirements.txt

`pip install -r requirements.txt`

# Dataset: 

We have used Smiles dataset which contains `13,165` grayscale images in the dataset, with each image having a size of `64×64` pixels from which `9,475` of these examples are not smiling while only `3,690` belong to the smiling class

# Train Model

We used `train_model.py` to train the network. This file takes two command line arguments --dataset is the path to the SMILES directory residing on disk and --model is the path to where the serialized LeNet weights will be saved after training

to run `train_model.py`, insert following command

`python train_model.py --dataset datasets/SMILEsmileD --model output/lenet.hdf5`

We used `detect_smile.py` to detect the face in videostream or web-cam to predict smiling or not-smiling at real time

To run detect_smile.py using your webcam, execute the following command

`python detect_smile.py --cascade haarcascade_frontalface_default.xml --model output/lenet.hdf5`

If you instead want to use a video file  you would update your command to use the --video switch

`python detect_smile.py --cascade haarcascade_frontalface_default.xml --model output/lenet.hdf5 --video path/to/your/video.mov`
