
# Le lien de l'API déployée sur heroku : https://mb-oc-p7-api.herokuapp.com/

import flask
from flask import Flask
from flask import request, jsonify
import joblib
import pandas as pd
import shap
import json
import numpy as np



app = Flask(__name__)
app.config["DEBUG"] = True

#PATH = './data/final/'

df = pd.read_pickle("df.gz")
print('df shape = ', df.shape)

#PATH = './data/preprocessed_data/'
#df = pd.read_csv(PATH+'df_final.csv')


#df = pd.read_pickle("df.gz")
#print('df shape = ', df.shape)

#df = pd.read_csv('df2.csv', nrows=200)
#print('data_test shape = ', df.shape)

#df = pd.read_csv('df4.csv')
#print('df shape = ', df.shape)

#df = pd.read_csv('data_test.csv', nrows=200)
#print('data_test shape = ', df.shape)

#PATH_1 = './data/preprocessed_data/'
#df = pd.read_csv(PATH_1+'data_global.csv', nrows=200)
#print('data_train shape = ', df.shape)



#Chargement du modèle
load_clf = joblib.load("model.joblib")
#explainer = joblib.load("shap_explainer.joblib")

#Premiers pas sur l'API
@app.route('/')
def index():
    return 'Welcome to my Flask API!'

@app.route('/score_min/', methods=['GET'])
def score_min():
    return {"score_min" : 0.55}

@app.route('/credit/<id_client>', methods=["GET"])
def credit(id_client):

    # Récupération des données du client en question
    ID = int(id_client)
    X = df[df['SK_ID_CURR'] == ID]
    
    print('identifiant du client :', ID)


    #Isolement des features non utilisées
    ignore_features = ['Unnamed: 0', 'SK_ID_CURR', 'INDEX', 'TARGET']
    relevant_features = [col for col in df.columns if col not in ignore_features]
    X = X[relevant_features]
    
    print('X shape = ', X.shape)
    
    #Imputation
    #imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    #X[X==np.inf] = np.nan
    #imputer.fit(X)
    #X_preprocess = imputer.transform(X)
    #Normalisation
    #scaler = StandardScaler()
    #scaler.fit(X_preprocess)
    #X_norm = scaler.transform(X_preprocess)
    
    proba = load_clf.predict_proba(X)
    print('Probabilité : ',proba)
    prediction = load_clf.predict(X)

    pred_proba = {
        'prediction': int(prediction),
        'proba': float(proba[0][0])
    }

    print('Nouvelle Prédiction : \n', pred_proba)

    return jsonify(pred_proba)
    #return json.dumps(pred_proba)


# Lancement de l'application
if __name__ == '__main__':
    #app.run(debug=True)
    app.run(port=1230, debug=True)
