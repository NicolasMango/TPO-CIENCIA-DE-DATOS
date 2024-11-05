from flask import Flask, request , render_template ,jsonify 
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
model = joblib.load("knn_model.pkl")
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/',methods=['GET'])
def form():
    return render_template('formv3.html')

@app.route('/predict',methods=['POST'])
def predict():
    data = request.form
    print(data)
    genre = data['genre_encoded']
    hours = data['Single-Player_Main Story_Average']
    review = 100
    print(genre)
    new_genre_encoded = label_encoder.transform([genre])[0]
    print(new_genre_encoded)
    features = np.array([[new_genre_encoded, hours,review]])
    print(features)
    #prediction = model.predict(pd.DataFrame(features))
    #prediction = model.kneighbors(pd.DataFrame(features), n_neighbors=5)
    # Obtener las 5 recomendaciones más cercanas
    distances, indices = model.kneighbors(pd.DataFrame(features), n_neighbors=5)
# Obtener los nombres de los juegos recomendados como un array
    prediction = [model.game_names[i] for i in indices[0]]  # y_train contiene los nombres de los juegos
    return render_template("formv3.html" , prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)