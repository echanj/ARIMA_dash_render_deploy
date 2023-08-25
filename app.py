import dash
from dash import Dash, html, dcc, callback, Output, Input
from callbacks import fit_arima_model
from callbacks import update_plot
from callbacks import create_ACF_plots


# Create Dash app
app = Dash(__name__)
server = app.server

# Define layout
app.layout = html.Div([
    html.H1("ARIMA Forecasting example using Plotly Dash"),

  # Steps with line breaks and styling
    html.H3("Steps (Do this in the following order to avoid errors !!!)",
            style={'whiteSpace': 'pre-wrap', 'text-align': 'left', 'margin': '20px', 'padding': '10px'}),
    html.P("1. Input ticker symbol and model parameters.",
           style={'whiteSpace': 'pre-wrap', 'text-align': 'left', 'margin-left': '30px'}),
    html.P("2. Click the 'Fit ARIMA Model' button and review the results" ,
           style={'whiteSpace': 'pre-wrap', 'text-align': 'left', 'margin-left': '30px'}),
    html.P("3. Adjust desired number of lags then click the 'Plot ACF and PACF' button. Use these plots to help adjust the desirable AR and MA.",
           style={'whiteSpace': 'pre-wrap', 'text-align': 'left', 'margin-left': '30px'}),
    html.P("4. Adjust number of Days to forecast then click the 'Plot ARIMA' button. HAVE FUN !!!",
           style={'whiteSpace': 'pre-wrap', 'text-align': 'left', 'margin-left': '30px'}),
    html.P("WARNING !!! This application is was not intended to be source of financial advice. Any investment decisions using the information gathered from this web based tool comes with great risk of loss !!!",
           style={'whiteSpace': 'pre-wrap', 'text-align': 'left', 'margin-left': '30px'}),


    html.Br(),html.Hr(), html.Br(), 
    # Input elements (e.g., date range picker)
    # Add your input components here
    # Text input for symbol string

    html.Div(className='row', 
             children=[
             'Input ticker symbol: ',
             dcc.Input(
                  id='symbol-input',
                  type='text',
                  placeholder='Enter symbol', debounce=True,
                  value='GOOG'  # Default symbol value
                  )],
    style={'textAlign': 'left', 'color': 'blue', 'fontSize': 30, 'margin': '10px', 'padding': '5px'}
    ), html.Br(),
    # Date range picker
    html.Div(className='row', 
             children=['Input date Range: ',
             dcc.DatePickerRange(
             id='date-picker-range',
             start_date='2023-01-01',
             end_date='2023-08-31',
             display_format='YYYY-MM-DD')],
             style={'textAlign': 'left', 'color': 'brown', 'fontSize': 20,'margin': '10px', 'padding': '5px'} ),
     html.Br(),

     html.Div([
         
        html.Div([
        html.Label('p-value:'),
        dcc.Input(id='p-value', type='number', placeholder='Enter p-value', value=2, debounce=True),
    ], style={'display': 'inline-block', 'margin-right': '10px'}),
    html.Div([
        html.Label('d-value:'),
        dcc.Input(id='d-value', type='number', placeholder='Enter d-value', value=1, debounce=True),
    ], style={'display': 'inline-block', 'margin-right': '10px'}),
    html.Div([
        html.Label('q-value:'),
        dcc.Input(id='q-value', type='number', placeholder='Enter q-value', value=2, debounce=True),
    ], style={'display': 'inline-block', 'margin-right': '10px'}),
    html.Div([
        html.Label('Test Set Length:'),
        dcc.Input(id='test-set-length', type='number', placeholder='Enter test set length', value=20, debounce=True),
    ], style={'display': 'inline-block', 'margin-right': '10px'}),
    html.Button('Fit ARIMA Model', id='fit-model-button'  ),
    html.Div(id='output')
     ]),

    html.Br(),html.Hr(), 
    # Display model results as text - improve display later 
    html.Div(id='model-summary-text',
             children='!!! Select parameters and click on \'fit ARIMA\' to generate model !!! ',
             style={'whiteSpace': 'pre-wrap','text-align': 'left', 'margin': '20px', 'padding': '10px' }), 
    html.Br(),html.Hr(), 

    html.Div([
           html.Button('Plot ACF and PACF', id='ACF-button'),  # Button to trigger ACF
           html.Div([html.Label('Number of lags:'), 
                     dcc.Input(id='lags', type='number', placeholder='nlags', value=20, debounce=True)], 
                     style={'display': 'inline-block', 'margin': '20px'}),
           dcc.Checklist(id='checkbox',options=[{'label': 'use training set', 'value': 'toggle'}],value=[]  # Initially, no value is selected
                   ),
           dcc.Graph(figure={}, id='ACF-plot') ]),  # Graph component for displaying the plot 
    

    html.Br(),html.Hr(), 
    html.Div([html.Button('Click here to plot Close and ARIMA predict, forecast, CI', id='plot-model-button'),  # Button to trigger model ploting
           html.Div([html.Label('Number of days to forecast into the future:'), 
                     dcc.Input(id='forecast-offset', type='number', placeholder='forecast days', value=5, debounce=True)], 
                     style={'display': 'inline-block', 'margin-left': '20px'}),
    dcc.Graph(figure={}, id='forecast-plot') ]),  # Graph component for displaying the plot

])

if __name__ == '__main__':
    app.run_server(debug=False)
