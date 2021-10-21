import pandas as pd
import numpy as np
import plotly.express as px
import requests
import pymongo
from dotenv import load_dotenv
from dash import Dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input,Output,State, ALL
import os

load_dotenv()

user=os.environ.get('USER')
password=os.environ.get('PASSWORD')
server=os.environ.get('SERVER')
bd=os.environ.get('DB')

geojson = requests.get('https://cdn.buenosaires.gob.ar/datosabiertos/datasets/barrios/barrios.geojson').json()
uri = f'mongodb+srv://{user}:{password}@{server}/{bd}?retryWrites=true&w=majority&ssl=true&ssl_cert_reqs=CERT_NONE'
data = list(pymongo.MongoClient(uri)['bohemio']['datos_barrio'].find())
df_scores = pd.DataFrame.from_records(data)

#defino un diccionario que contiene las etiquetas de cada indicador.
score_labels = {
    'score_poblacion':'Población',
    'score_valuacion':'Valuación',
    'score_educacion':'Educación',
    'score_salud':'Salud',
    'score_transporte':'Accesibilidad/Transporte',
    'score_verde':'Espacios Verdes',
    'score_delitos':'Seguridad',
    'score_accidentes':'Accidentes',
    'score_esparcimiento':'Esparcimiento',
    'score_barriospop':'Barrios Populares',
}

score_labels_inv = {
    'score_poblacion':'Población',
    'score_valuacion_inv':'Valuación',
    'score_educacion':'Educación',
    'score_salud':'Salud',
    'score_transporte':'Accesibilidad',
    'score_verde':'Espacios Verdes',
    'score_delitos_inv':'Seguridad',
    'score_accidentes_inv':'Accidentes',
    'score_esparcimiento':'Esparcimiento',
    'score_barriospop_inv':'Barrios Populares',
}

#listado de scores
data = ['area','comuna', 'poblacion', 'valorxm2', 'cant_escuelas', 'cant_hospitales', 'cant_bicis', 'cant_estaciones', 'm2verdes', 'cant_comis', 'ind_delincuencia', 'cant_bp']
data_mult = ['BARRIO']+data

app = Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server
app.title='Bohemio - Conocé tu Ciudad'


navbar=dbc.Navbar([
        html.A(
            dbc.Row(
                [
                    dbc.Col(html.Img(src='https://i.ibb.co/TWFfpmG/Group-2.png', height="60rem")),
                ],
                align="center",
            )
        ),
        html.Div(
            [
             dbc.NavItem(dbc.NavLink("Investigá", href="/page-1", style={'text-decoration':'None', 'color':'#0E4666'})),
             dbc.NavItem(dbc.NavLink("Buscá", href="/page-2", style={'text-decoration':'None', 'color':'#0E4666'})),
             dbc.NavItem(dbc.NavLink("Informate", href="/page-3", style={'text-decoration':'None', 'color':'#0E4666'}))
            ],className="g-0 ps-5", style={'display':'flex'}
        )
        
    ],
    color='#FFFFF',
    className='pt-4'
    )

indicadores = dbc.Card([
                        html.H5('Seleccionar Indicador', style={'color':'#0E4666'}),
                        dbc.Checklist(id='indicador', options=[{'value':i,'label':score_labels_inv.get(i)} for i in score_labels_inv], value=[],className='pt-2', style={'color':'#0E4666'})
              ],body=True)

sliders = dbc.Card(
                   children=[], className='pt-4 pb-4', id='card', style={'color':'#0E4666'}
                   )



checklist = dbc.Card([
                html.H5('Seleccionar indicador', style={'color':'#0E4666'}),
                html.P('Agregue todos los indicadores que desee relacionar y visualizar en el Mapa.', style={'color':'#0E4666'}),
                dbc.Checklist(id='seleccion_score', options=[{'value':i,'label':score_labels.get(i)} for i in score_labels],
                              value=list(score_labels)[:1], labelStyle={'display':'inline-block','color':'#0E4666'})
            ], body=True)   


dropdown = html.Div([
                     dbc.Row(dbc.Col(html.H5('Selecciona el barrio'), style={'color':'#0E4666'}, md=12)),
                     dbc.Row([
                              dbc.Col(
                                dcc.Dropdown(id='dropdown-1', options=[{'value':i,'label':i.title()} for i in df_scores.BARRIO], style={'color':'#0E4666'}, value='PALERMO')
                              ),
                     ], className='pt-3')
                     
])

page_1 = dbc.Container([
    navbar,
    html.Hr(),
    dbc.Row([
        dbc.Col([
                checklist    
        ], md=4, sm=12),
        dbc.Col([
            dbc.Card([
                html.H5('Barrios de CABA', style={'color':'#0E4666'}),
                dcc.Graph(id='fig1')
            ],body=True, style={"width": "40rem"})
        ], md=8, sm=12)
    ]),
])


page_2 = dbc.Container([
    navbar,
    html.Hr(),
    dbc.Row([
        dbc.Col(indicadores, md=2),
        dbc.Col(sliders, md=4,className='text-center'),
        dbc.Col(
                 [
                  dbc.Card([ 
                      dcc.Graph(id='fig2')])
                  ], md=6)
    ]),
])


page_3 = dbc.Container([
                        navbar,
                        html.Hr(),
                        dropdown,
                        dbc.Row(
                            dbc.Card([
                                  dcc.Graph(id='fig3')
                        ]), className='pt-4'
                        )
                        
])

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content', children=page_1)
])

@app.callback(
    Output('fig1', 'figure' ),
    Input('seleccion_score','value')
)
def display_grafico(seleccion_score):
    
    if len(seleccion_score) == 1:
            df_scores['score_cat'] = df_scores[seleccion_score[0]].astype(str) 
            fig = px.choropleth(df_scores, geojson=geojson, color='score_cat',
                                locations="BARRIO", featureidkey="properties.BARRIO",
                                projection="mercator", color_discrete_sequence= ['#7AC45C', '#AEEB5A','#73D2BF','#FC5845', '#C40186']
                                ,category_orders={'score_cat':['5','4','3','2','1']}, labels = {'score_cat': 'Ind. Resultado'}, hover_data=data
                               )
            fig.update_geos(fitbounds="locations", visible=False)
            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    
    elif len(seleccion_score) > 1:
            df_dinamico = df_scores.loc[:, data_mult+seleccion_score]
            df_dinamico['suma_scores'] = df_dinamico[seleccion_score].sum(axis = 1, numeric_only=True)
            df_dinamico['score_resultado'] = pd.qcut(df_dinamico['suma_scores'], q = 5, labels=['1','2','3','4','5'])
     
            fig = px.choropleth(df_dinamico, geojson=geojson, color=df_dinamico.score_resultado,
                                locations="BARRIO", featureidkey="properties.BARRIO",
                                projection="mercator", color_discrete_sequence= ['#7AC45C', '#AEEB5A','#73D2BF','#FC5845', '#C40186'],
                                category_orders={'score_resultado':['5','4','3','2','1']}, labels = {'score_resultado':'Ind. Resultado'}, hover_data=data 
                                )
            fig.update_geos(fitbounds="locations", visible=False)
            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    else:
      fig={}
    return fig

@app.callback(
    Output('card', 'children'),
    Input('indicador', 'value')
)
def sliders(indicador):
  if len(indicador) > 0:
    labels = [dbc.Label(id=i, children=i.split('_')[1].capitalize()) for i in indicador]
    sliders = [dcc.RangeSlider(id={'type':'my-slider','index':i}, min=1, max=5, step=1, value=[1,2], marks={1:'Muy Bajo', 2:'Bajo',3:'Normal',4:'Alto',5:'Muy Alto'}, className='ml-3 mr-3',persistence=True) for i in indicador]
    combined = []
    for i in range(len(labels)):
      combined.append(labels[i])
      combined.append(sliders[i])
  else:
    combined = []

  return combined
  

@app.callback(
    Output('fig2', 'figure'),
    Input({'type': 'my-slider', 'index': ALL}, 'id'),
    Input({'type': 'my-slider', 'index': ALL}, 'value')
)
def display(ids, values):
  try:
    sliders = [[ids[i]['index'], values[i]] for i in range(len(ids))]
    df = df_scores.loc[(df_scores[sliders[0][0]] >= sliders[0][1][0]) & (df_scores[sliders[0][0]] <= sliders[0][1][1])]
    data = []
    for i in (sliders):
      data.append(i[0])
      df = df.loc[(df[i[0]] >= i[1][0]) & (df[i[0]] <= i[1][1]), :]

    df_scores['Resultado'] = np.where(df_scores.BARRIO.isin(df.BARRIO), 'Cumple Requisitos', 'No Cumple Requisitos')

    fig = px.choropleth(df_scores, geojson=geojson, color='Resultado',
                      locations="BARRIO", featureidkey="properties.BARRIO",
                      projection="mercator", hover_data=data, color_discrete_map={'Cumple Requisitos': '#245dd6', 'No Cumple Requisitos': '#849dd1'}
                    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig  
  except:
    return {}


@app.callback(
    Output('fig3', 'figure'),
    Input('dropdown-1', 'value')
)
def polar(val1):
  if val1 is not None:
    df_drop = df_scores.loc[df_scores.BARRIO == (val1)][score_labels_inv.keys()].transpose().reset_index()
    df_drop.columns = ["label","valor"]
    df_drop["label"] = [i.split("_")[1].capitalize() for i in df_drop["label"]]
    fig = px.line_polar(df_drop,r=df_drop.valor, theta=df_drop.label,width=500,line_close=True)
  else:
    fig={}
  return fig


@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/page-1':
        return page_1
    elif pathname == '/page-2':
        return page_2
    elif pathname == '/page-3':
        return page_3
    else:
        return page_1

if __name__ == '__main__':
    app.run_server()
