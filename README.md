# Ensemble Models
<p align="center">
  <img width="460" height="300" src="https://miro.medium.com/max/600/1*ZzXwFueV-Beh9MapLgZ5QA.png">
</p>

In statistics and machine learning, ensemble methods use multiple learning algorithms to obtain better predictive performance than could be obtained from any of the constituent learning algorithms alone.


# Project Description

In this project, four predictive machine learning models are applied to one dataset in order to select the model that returns the most accurate predictions. Different values for the hyperparameters *max_depth* and *n_estimators* were used in order to optimize model perforamnce by obtaining the highest accuracy and AUC scores. Lastly, the best-performing model was selected by comparing the scores when using common hyperparameters.


## Steps

 1. Splitting Data into Train and Test Sets
 2. Fitting the Model
 3. Calculating and Graphing Accuracy Scores
 4. Comparing Performances

## Requirements

**Python.** Python is an interpreted, high-level and general-purpose programming language. 

**Integrated Development Environment (IDE).** Any IDE that can be used to view, edit, and run Python code, such as:
- [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true)
- [Jupyter Notebook](https://jupyter.org/).

### Packages 
Install the following packages in Python prior to running the code.
```python
!pip install xgboost

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

drive.mount('/content/drive')
```
After importing ```drive.mount('/content/drive')```, follow instructions in the output to authorize access to Google Drive in order to obtain directories.

## Launch
Download the Python File *CA04_Ensemble_Models* and open it in the IDE. Download and import the dataset *data.csv*. 

## Authors

[Silvia Ji](https://www.linkedin.com/in/silviaji/) - [GitHub](github.com/jisilvia)

## License
This project is licensed under the [MIT](https://choosealicense.com/licenses/mit/) License.

## Acknowledgements

The project template and dataset were provided by [Arin Brahma](https://github.com/ArinB) at Loyola Marymount University.
