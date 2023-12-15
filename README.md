# Final Project - Tumor classification model
open sw final project

tumor classification model

## Project explanation
This project uses Scikit-Learn to create a machine learning model to classify tumors

## Build status
- python 3.11.5
- numpy : 1.24.4
- scikit-learn: 1.3.2
- scikit-image : 0.21.0

## Training dataset
### 1. glioma_tumor
![gg (1)](https://github.com/Jongwon822/Jongwon_Final/assets/147024868/c926ec7c-235c-48ff-9110-7f8ecf071ace)
### 2. meningioma_tumor
![m (2)](https://github.com/Jongwon822/Jongwon_Final/assets/147024868/10af03d9-13d2-4223-aa0c-483e141720af)
### 3. no_tumor
![1](https://github.com/Jongwon822/Jongwon_Final/assets/147024868/a7c7171d-a4ce-4615-85a9-bfe56824e0dc)
### 4. pituitary_tumor
![p (1)](https://github.com/Jongwon822/Jongwon_Final/assets/147024868/886e39bc-6132-454b-a903-d89ff2ed4bb4)

## Model training and prediction

### 1. Load Packages
```python
import os

import sklearn.datasets
import sklearn.linear_model
import sklearn.svm
import sklearn.tree
import sklearn.ensemble
import sklearn.model_selection
import sklearn.metrics
import sklearn.neighbors
import sklearn.preprocessing

import skimage.io
import skimage.transform
import skimage.color

import numpy as np

import matplotlib.pyplot as plt 
%matplotlib inline

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
```

### 2. Load Data Points
```python
image_size = 64
labels = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']

images = []
y = []
for i in labels:
    folderPath = os.path.join('./tumor_dataset/Training',i)
    for j in os.listdir(folderPath):
        img = skimage.io.imread(os.path.join(folderPath,j),)
        img = skimage.transform.resize(img,(image_size,image_size))
        img = skimage.color.rgb2gray(img)
        images.append(img)
        y.append(i)

images = np.array(images)

X = images.reshape((-1, image_size**2))
y = np.array(y)
```

### 3. Data Division
```python
X_train, X_test, y_train, y_test
= sklearn.model_selection.train_test_split(X, y, test_size=0.015, random_state=0)
```
##### Variable Explanation
- 'X_train' is feature vectors of training dataset
- 'y_train' is target labels of training dataset
- 'X_test' is feature vectors of test dataset
- 'y_test' is target labels of test dataset
- 'y_pred' was initialized as zero vectors and fill 'y_pred' with predicted labels

### 4. Model training
#### KNeighbors
```python
knn = KNeighborsClassifier(n_neighbors= 1,metric='minkowski',p=2,weights='distance',n_jobs=-1)
```

#### SVC
```python
svm = SVC(C=1,gamma=1,kernel='rbf',random_state=1234)
```

#### Extra Tree
```python
etc = ExtraTreesClassifier(n_estimators = 200,max_features = 2,
                           max_depth = 30,min_samples_split = 3,random_state = 1034,n_jobs = -1)
```

#### Ridge
```python
rgc = RidgeClassifier(alpha = 0.5,class_weight='balanced')
```

#### Voting
```python
vote = VotingClassifier(estimators = [
    ('etc', etc),
    ('knn', knn),
    ('rgc', rgc),
    ('svm', svm)
], n_jobs=-1)
```

#### Voting classifier training
```python
vote.fit(X_train, y_train)
y_pred = vote.predict(X_test)
```

### 5. Accuracy
```python
print('Accuracy: %.2f' % sklearn.metrics.accuracy_score(y_test, y_pred))
```

## Accuracy Results
accuracy : 86% (My test)

## author
name : Lee Jong Won

e-mail : ljwon0415@naver.com

## reference
https://scikit-learn.org/stable/user_guide.html

https://scikit-learn.org/stable/modules/classes.html
