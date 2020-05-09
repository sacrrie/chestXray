# Project Chest X-ray Diagnosis Code Running Instructions
## Open for review purpose only

Third party libraries requirements:
+ numpy
+ matplotlib
+ Pytorch
+ pandas
+ scikit-learn

Please install them before running the script.
```bash
pip install <required-lib>
```
or using Conda
```bash
conda install <required-lib>
```
Download the NIH Chest X-ray dataset and extract to the images folder, the link is below:
https://nihcc.app.box.com/v/ChestXray-NIHCC

And run the script with Python 3.

Run instance:

Preprocessing the labels and produce train, validation and test sets
```bash
python preprocessing.py
```
Train the model and the best iteration would be saved
```bash
python train.py
```
Test the best model with the test dataset
```bash
python test.py
```

It will take sometime to finish running depends on your computer specification.
After the script finished running. Resuls will appear in the terminal. 

*Thank you for your effort reviewing my project, have a good day!*


__Reference__
[Zoogzog/Chexnet](github.com/zoogzog/chexnet)
[Arnoweng/CheXNet](github.com/arnoweng/CheXNet)

