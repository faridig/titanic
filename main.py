from flask import Flask, request, jsonify
from fonction_custom import extract_cabin_letter
import sklearn
import pandas as pd
import numpy as np
import joblib
from io import StringIO   

# Load the model and the pipeline
pipeline = joblib.load('/home/faridig/dev_ia/titanic/titanic.model.pkl')
 
#démarrer l'application flask
app = Flask(__name__)

# faire des prédictions

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convertir le JSON en DataFrame
        data = request.json
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
        
        # Faire des prédictions
        resultat = pipeline.predict(df)[0]
        
        #afficher le résultat
        print(f"Résultat : {resultat}", flush=True)
        
        return ({'prediction': str(resultat)}), 201
        
    except Exception as e:
        return str(e), 400

        
   
 
#page d'accueil

@app.route('/')
def index():
    return '<h1>API Titanic. Utiliser /predict en POST pour faire des prédictions sur le Titanic</h1>'

#si on est le main on lance l'application
if __name__ == '__main__':
    app.run(debug=True)