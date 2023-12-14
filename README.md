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
```

### 2. Load Data Points
```python
# Data augmentation function definition
def augment_image(image):
    # symmetry
    if np.random.rand() < 0.5:
        image = np.fliplr(image)
    
    # Random rotation (-45 to 45 degrees)
    angle = np.random.uniform(-45, 45)
    image = skimage.transform.rotate(image, angle, mode='reflect', preserve_range=True)
    
    return image

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
        
        # Add augmented image
        augmented_img = augment_image(img)
        images.extend([img, augmented_img])
        y.extend([i, i]) 
        
        
images = np.array(images)

X = images.reshape((-1, image_size**2))
y = np.array(y)
```




### 3. Data preprocessing
```python
X_train, X_test, y_train, y_test
= sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=0)

scaler = sklearn.preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
##### Variable Explanation
- 'X_train' is feature vectors of training dataset
- 'y_train' is target labels of training dataset
- 'X_test' is feature vectors of test dataset
- 'y_test' is target labels of test dataset
- 'y_pred' was initialized as zero vectors and fill 'y_pred' with predicted labels

### 4. Model training
#### KNN
```python
knn1 = sklearn.neighbors.KNeighborsClassifier(n_neighbors=2,metric='minkowski',
                                              p=2,weights='distance',n_jobs=-1)
knn2 = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3,metric='minkowski',
                                              p=1,weights='distance',n_jobs=-1)
```

#### SVM
```python
svm1 = sklearn.svm.SVC(C=50,gamma=0.01,kernel='rbf',
                       random_state=100,probability=True)
```

#### RandomForest
```python
rfc1 = sklearn.ensemble.RandomForestClassifier(n_estimators=200,max_depth=25,
                                               min_samples_split=8,n_jobs=-1)
rfc2 = sklearn.ensemble.RandomForestClassifier(max_depth=25, max_features='sqrt', min_samples_leaf=1, 
                                               min_samples_split=5, n_estimators=200)
```

#### Voting
```python
vote = sklearn.ensemble.VotingClassifier(estimators=[
    ('knn1', knn1),
    ('knn2', knn2),
    ('svm1', svm1),
    ('rfc1', rfc1),
    ('rfc2', rfc2)
], voting='soft',n_jobs=-1)
```

#### Voting classifier training
```python
vote.fit(X_train_scaled, y_train)
y_pred = vote.predict(X_test_scaled)
```

### 5. Accuracy
```python
print('Accuracy: %f' % sklearn.metrics.accuracy_score(y_test, y_pred))
```

## Accuracy Results
accuracy : 79% (Professor's test)

accuracy : 91% (My test)

## author
name : LEE JONG WON

e-mail : ljwon0415@naver.com

## reference
https://scikit-learn.org/stable/user_guide.html

https://scikit-learn.org/stable/modules/classes.html
