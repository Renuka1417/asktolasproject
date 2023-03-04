# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 21:16:03 2023

@author: Renuka
"""

import numpy as np
import pickle
import streamlit as st

with open('E:/ml model deployment/nlp_model.pkl','rb') as f:
 cv, classifier= pickle.load(f)
#create function for prediction
def sentiment_analysis(review):
    X_test = [review]
    X_test_vect = cv.transform(X_test)
    y_pred = classifier.predict(X_test_vect)
    print(y_pred)


    if y_pred[0] == 1:
        return "Positive review"
    else:
        return "Negative review"
    
    
def main():
    
    #title of page
    st.title("REVIEW ANALYSIS APP")
    

    review = st.text_input("Enter a review:")
    if review:
      result = sentiment_analysis(review)
      st.write("Sentiment:", result)
    
    
if __name__ == '__main__':
    main()
    #to run program run following code in promt
   # streamlit run "E:\ml model deployment\app.py"
