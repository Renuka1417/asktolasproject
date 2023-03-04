# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle
with open('C:\Users\Renuka\nlp_model.pkl','rb') as f:
 cv, classifier= pickle.load(f)
X_test = ["This is a neutral review."]
X_test_vect = cv.transform(X_test)
y_pred = classifier.predict(X_test_vect)
print(y_pred)


if y_pred[0] == 1:
    print("Negative review")
else:
    print("Positive review")
