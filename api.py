from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np

# Configuración de la aplicación Flask
app = Flask(__name__)

# Cargar el modelo y el encoder
model = joblib.load("knn_model.pkl")
label_encoder = joblib.load('label_encoder.pkl')

# Ruta principal de Flask
@app.route('/', methods=['GET'])
def form():
    return render_template('formv3.html')

# Ruta para hacer la predicción
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    genre = data['genre_encoded']
    hours = data['Single-Player_Main Story_Average']
    review = 100
    
    # Codificar el género y preparar las características
    new_genre_encoded = label_encoder.transform([genre])[0]
    features = np.array([[new_genre_encoded, hours, review]])
    
    # Obtener las 5 recomendaciones más cercanas
    distances, indices = model.kneighbors(pd.DataFrame(features), n_neighbors=5)
    prediction = [model.game_names[i] for i in indices[0]]
    
    return render_template("formv3.html", prediction=prediction)

# Iniciar la aplicación
if __name__ == '__main__':
    app.run(debug=True)
