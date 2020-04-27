import pandas as pd
from sklearn import datasets 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score 
from sklearn.neighbors import KNeighborsClassifier 
import pickle 

df = pd.read_excel('C:/Users/SaiKrishna/Edu/Project/DataSet/Rohit_Sharma-Data.xlsx') 
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
validation_size = 0.20 
seed = 100 
X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=validation_size, 
                                                    random_state=seed) 

svc = SVC(kernel = 'linear')
svc.fit(X_train,Y_train)

with open('models/rohit_classifier_model.pk', 'wb') as model_file: 
    pickle.dump(svc, model_file)