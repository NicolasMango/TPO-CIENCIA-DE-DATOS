from flask import Flask, request , render_template ,jsonify 
import joblib
import pandas as pd
import numpy as np
app = Flask(__name__)
model = joblib.load("knn_model.pkl")
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/',methods=['GET'])
def form():
    return render_template('form.html')

@app.route('/predict',methods=['POST'])
def predict():
    data = request.form
    print(data)
    genre = data['genre_encoded']
    hours = data['Single-Player_Main Story_Average']
    print(genre)
    new_genre_encoded = label_encoder.transform([genre])[0]
    print(new_genre_encoded)
    features = np.array([[new_genre_encoded, hours]])
    print(features)
    prediction = model.predict(pd.DataFrame(features))
    return render_template("form.html" , prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)