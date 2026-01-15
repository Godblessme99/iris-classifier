\# Iris Classifier (Decision Tree)





\## Overview

This project is an end-to-end machine learning example built as part of the

AI Fundamentals module. It trains a Decision Tree classifier on the classic

Iris dataset using scikit-learn.

The project demonstrates the full workflow of supervised learning:

data loading, train/test splitting, model training, evaluation, and result

visualisation.





\## Project Structure



```

iris-classifier/

├── data/                # empty (Iris loaded from scikit-learn)

├── notebooks/

│   └── iris\_model.ipynb # walk-through notebook

├── src/

│   └── train.py         # reproducible CLI script

├── outputs/

│   └── confusion\_matrix.png

├── tests/

├── requirements.txt

└── README.md

```





\## How to Run



```bash

git clone https://github.com/Godblessme99/iris-classifier.git

cd iris-classifier

python -m venv venv

venv\\Scripts\\activate

pip install -r requirements.txt

python src/train.py --test-size 0.2 --random-state 42    

```





\## Results

The Decision Tree classifier achieved perfect accuracy on the test set for this

train/test split. The confusion matrix shows correct classification for all

three iris species.











