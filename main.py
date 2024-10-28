import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


st.title('Multiple classifier for different Dataset')



dataset_name = st.sidebar.selectbox('Select your Dataset',('Iris','Breast cancer','Wine'))

classifier_name = st.sidebar.selectbox('Select your classifier',('KNN','SVM','Random forest'))


def get_dataset(dataset_name):
    if dataset_name == 'Iris':
       data=datasets.load_iris()
    elif dataset_name == 'Breast cancer':
        data = datasets.load_breast_cancer() 
    else :
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return data

data = get_dataset(dataset_name)
df = pd.DataFrame(data.data, columns=data.feature_names)
X = data.data
y = data.target
class_names = data.target_names



st.write('shape of the dataset',X.shape)
st.write('Number of classes of the data set',len(np.unique(y)))

#Analyse exploratoire


x_axis=st.sidebar.selectbox('Select the feaure for the X Axis',data.feature_names)
y_axis=st.sidebar.selectbox('Select the feaure for the Y Axis',data.feature_names)
fig, ax = plt.subplots()
plt.scatter(df[x_axis],df[y_axis],c=y,alpha=0.5,cmap='viridis')
plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.colorbar()

st.pyplot(fig)

def add_parametre(clf_name):
    params = dict()
    if clf_name == 'KNN':
        K = st.sidebar.slider('K',1,15)
        params['K']=K
    elif clf_name == 'SVM':
        C = st.sidebar.slider('C',0.01,10.0)
        params['C']=C
    else:
        max_depth = st.sidebar.slider('max_depth',2,15)
        n_estimators = st.sidebar.slider('n_estimators',1,100)
        params['max_depth']=max_depth
        params['n_estimators']=n_estimators
    return params

params=add_parametre(classifier_name)

def get_classifier(clf_name,params):
    if clf_name=='KNN':
        clf=KNeighborsClassifier(n_neighbors=params['K'])
    elif clf_name=='SVM':
        clf=SVC(C=params['C'])
    else :
        clf=RandomForestClassifier(n_estimators=params['n_estimators'],max_depth=params['max_depth'])
    return clf

clf = get_classifier(classifier_name,params)

#Classification
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=6)

clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

acc=accuracy_score(y_test,y_pred)
st.write(f'Classifier ={classifier_name}')
st.write(f'Accuracy = {acc}')

#Plot
pca = PCA(2)
X_projected=pca.fit_transform(X)
x1=X_projected[:,0]
x2=X_projected[:,1]

fig, ax = plt.subplots()
plt.scatter(x1,x2,c=y,alpha=0.5,cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

st.pyplot(fig)

def get_values_to_predict(dataset_name):
    values_array=[]
    for columns in df.columns:
        values = st.number_input(f"Please insert {columns} value",min_value=0.0,step=0.1,max_value=5.0)
        values_array.append(values)
    return(values_array)



values_array=get_values_to_predict(dataset_name)
values_array_np=np.array(values_array)
values_array=values_array_np.reshape(-1,1)

predict = ''
def get_prediction(values_array):
    global predict
    predict_index=clf.predict(values_array.reshape(1, -1))
    predict = class_names[predict_index]
    return predict
     
    

if st.button('Prédire'):
    get_prediction(values_array)
    
# Afficher la prédiction sous le bouton
if predict is not None:  # Vérifie si predict a une valeur
    #st.write(f'Prédiction : {predict[0]}')    
    st.write(f'The prediction for these valuse is {predict}')