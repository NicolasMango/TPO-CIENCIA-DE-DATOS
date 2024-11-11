from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
from dash import Dash, dcc, callback, dash_table, html
from dash.dependencies import Output, Input
import plotly.express as px
import dash_mantine_components as dmc

# Configuración de la aplicación Flask
app = Flask(__name__)

# Cargar el modelo y el encoder
model = joblib.load("knn_model.pkl")
label_encoder = joblib.load('label_encoder.pkl')

# Datos de ejemplo para Dash
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')

# Configurar Dash dentro de la aplicación Flask
app_dash = Dash(
    __name__,
    server=app,
    url_base_pathname='/dashboard/'  # Ruta específica para el dashboard
)

# Configurar el layout de Dash
app_dash.layout = dmc.Container([
    dmc.Title('Mi Primera App con Datos, Gráficos y Controles', color="blue", size="h3"),
    dmc.RadioGroup(
        [dmc.Radio(i, value=i) for i in ['pop', 'lifeExp', 'gdpPercap']],
        id='my-dmc-radio-item',
        value='lifeExp',
        size="sm"
    ),
    dmc.Grid([
        dmc.Col([
            dash_table.DataTable(
                data=df.to_dict('records'),
                page_size=12,
                style_table={'overflowX': 'auto'}
            )
        ], span=6),
        dmc.Col([
            dcc.Graph(figure={}, id='graph-placeholder')
        ], span=6),
    ]),
], fluid=True)

# Callback de Dash para actualizar el gráfico
@app_dash.callback(
    Output(component_id='graph-placeholder', component_property='figure'),
    Input(component_id='my-dmc-radio-item', component_property='value')
)
def update_graph(col_chosen):
    fig = px.histogram(df, x='continent', y=col_chosen, histfunc='avg')
    return fig

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
