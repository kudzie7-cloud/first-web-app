import dash
import dash_core_components as dcc
import dash_html_components as html
my_app = dash.Dash()
my_app.layout = html.Div(children= [html.H1(children= 'Hello, Dash!'),
                                     html.Div(children= '''
                                              Dash: A web application framework for python.'''),
                                              dcc.Graph(
                                              id = 'Example-Graph', 
                                              figure={
                                               'data': [
                                                {'x': [1, 2, 3], 'y': [4, 1, 2],
                                                 'type': 'bar', 'name': 'SF'},
                                                 {'x': [1, 2, 3], 'y': [2, 4, 5],
                                                  'type': 'bar', 'name': 'Montreal'}
                                               ],
                                               'layout': {
                                                'title': 'Dash data Visualisation'
                                               }
                                              }
                                              )
                                              ])
if __name__ == '_main_':
 my_app.run_server(debug = True)
