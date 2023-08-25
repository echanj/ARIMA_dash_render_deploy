# callbacks.py
from dash import Output, Input
from dash import callback
import yfinance as yf
import re
import numpy as np
import pandas as pd
import dash
from dash import Dash, html, dcc, callback, Output, Input
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from sklearn.model_selection import TimeSeriesSplit
import plotly.graph_objects as go
from plotly.subplots import make_subplots



# define helper functions
def get_price_data(symbol, start_date , end_date , interval='1d'):
    return yf.download(symbol, start=start_date, end=end_date, interval=interval)


@callback(
  #  Output('df_train-store', 'data'),  # Store model results as HTML in this Div
  #  Output('df_test-store', 'data'),  # Store model results as HTML in this Div
    Output('model-summary-text', 'children'),
    Input('fit-model-button', 'n_clicks'),  # Button to trigger model fitting
    [Input('p-value', 'value'), Input('d-value', 'value'), Input('q-value', 'value')],
        Input('symbol-input', 'value'),  # Add input elements here
        Input('date-picker-range', 'start_date'),
        Input('date-picker-range', 'end_date'),
        Input('test-set-length', 'value')
)

def fit_arima_model(n_clicks, p, d, q, symbol, start_date, end_date, test_size):
  #  global model
  #  global model_res
#    global df_train
    global df
#    global df_test 

    if n_clicks is None:
        return dash.no_update

    df = get_price_data(symbol, start_date=start_date, end_date=end_date, interval='1d')

    tss = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=2, test_size=test_size)

    for train_index, test_index in tss.split(df):
        df_train=df.Close[train_index]
        df_test=df.Close[test_index]

    # Fit ARIMA model with selected parameters
    model = sm.tsa.ARIMA(df_train, order=(p, d, q))
    model_res = model.fit(method_kwargs={"maxiter": 1000})

    # Display model results as text
    summary_text = model_res.summary().as_text()
    summary_text = re.sub(' +', '\t', summary_text)

    return summary_text

  #  return df_train.to_json(date_format='iso', orient='split'), df_test.to_json(date_format='iso', orient='split')


# Define callback function to update the plot
@callback([
    Output('forecast-plot', 'figure')],
    Input('plot-model-button', 'n_clicks'),  # Button to trigger model fitting
    [Input('p-value', 'value'), Input('d-value', 'value'), Input('q-value', 'value')],
        Input('symbol-input', 'value'),  # Add input elements here
        Input('test-set-length', 'value'), 
        Input('forecast-offset', 'value') 
)


#def update_plot(df_train_store,df_test_store):
def update_plot(n_clicks, p, d, q, symbol, test_size, offset):
    # Your data preprocessing and ARIMA model code here
    # Replace this with your data retrieval and processing code
    # GOOG_df = get_price_data('GOOG', start_date='2023-01-01', end_date='2023-08-31', interval='1d')
   # df = get_price_data(symbol, start_date=start_date, end_date=end_date, interval='1d')

    # global model

    if n_clicks is None:
         return dash.no_update

    tss = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=2, test_size=test_size)

    for train_index, test_index in tss.split(df):
        df_train=df.Close[train_index]
        df_test=df.Close[test_index]

    # Replace the ARIMA modeling code
    model = sm.tsa.ARIMA(df_train.asfreq('D'), order=(p, d, q))
    model_res = model.fit(method_kwargs={"maxiter": 1000})

    # df_train = pd.read_json(df_train_store, orient='split')
    # df_test = pd.read_json(df_test_store, orient='split')

    
# test codes for to possibly improve the display and further stylize the text 
    # Format model summary with HTML tags and CSS
   # summary_text = model_res.summary().as_html()
   # summary_text = summary_text.replace('\n', '').replace('<table>', '<table class>="table table-striped"')

    # Format model summary using re to remove unnecessary <p> tags
   # summary_html = model_res.summary().as_html()
   # summary_html = re.sub(r'<p(.*?)>', '', summary_html)
   # summary_html = re.sub(r'</p>', '<br>', summary_html)

   # offset=10

    pred_train = model_res.predict()
    fc = model_res.get_forecast(len(df_test.asfreq('D'))+offset).predicted_mean
    ci = model_res.get_forecast(len(df_test.asfreq('D'))+offset, alpha=0.05).conf_int()

    # Create the Plotly figure
    fig = make_subplots()
    
    # Add your traces (similar to your plot_ARIMA_results function)
    # Replace this with your code to create the Plotly traces
    trace1 = go.Scatter(x=df.index, y=df.Close, mode='lines', name='Close Price', showlegend=True)
    trace2 = go.Scatter(x=pred_train.index[1:], y=pred_train[1:], mode='lines', name='Predicted', showlegend=True)
    trace3 = go.Scatter(x=ci.index, y=ci['lower Close'], mode='lines', name='Lower CI', fillcolor='rgba(0, 0, 255, 0.2)', line=dict(color='rgba(255, 255, 255, 0)'), showlegend=False)
    trace4 = go.Scatter(x=ci.index, y=ci['upper Close'], fill='tonexty', mode='lines', name='CI bounds', fillcolor='rgba(0, 0, 255, 0.2)', line=dict(color='rgba(255, 255, 255, 0)'), connectgaps=False, showlegend=True)
    trace_fc = go.Scatter(x=fc.index, y=fc, mode='lines', name='forecast', showlegend=True)

    # Add your traces to the figure
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.add_trace(trace3, row=1, col=1)
    fig.add_trace(trace4, row=1, col=1)
    fig.add_trace(trace_fc)
    
    # Set layout options (similar to your code)
    fig.update_layout(
        title=f"ARIMA {model.order} with Forecast vs Actuals on {symbol} closing price data",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Price"),
        height=400,
        width=700
    )
    
    return [fig]

@callback([Output('ACF-plot', 'figure')],
    [Input('ACF-button', 'n_clicks'),
    Input('lags', 'value'),Input('checkbox', 'value')],
        Input('test-set-length', 'value') 
#        Input('symbol-input', 'value'), 
#        Input('date-picker-range', 'start_date'),
#        Input('date-picker-range', 'end_date')]
)

def create_ACF_plots(n_clicks,nlags,checkbox_value,test_size):

    if n_clicks is None:
         return dash.no_update

    use_training_set=True if 'toggle' in checkbox_value else False
    
    if use_training_set:
        tss = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=2, test_size=test_size)
        for train_index, test_index in tss.split(df):
            series=df.Close[train_index]

    else:
       series = df.Close   
    # sp_titles=("1st Order differencing, ACF, PACF ")
    fig = make_subplots(rows=3, cols=1, 
                        vertical_spacing=0.1,
                        horizontal_spacing=0.05)
    
    corr_array = acf(series.dropna(), nlags=nlags, alpha=0.05)
    lower_y = corr_array[1][:,0] - corr_array[0]
    upper_y = corr_array[1][:,1] - corr_array[0]
  

    
    fig.add_trace(go.Scatter(x=series.index, y=series.diff(),name='1st order diff',showlegend=True  ), row=1, col=1)

    [fig.add_trace(go.Scatter(x=(x,x), y=(0,corr_array[0][x]), mode='lines',line_color='#3f3f3f',showlegend=False), row=2, col=1) for x in range(len(corr_array[0]))]
    fig.add_trace(go.Scatter(x=[x for x in range(len(corr_array[0]))], y=corr_array[0],mode='markers', marker_color='#1f77b4',marker_size=8,name='ACF'  ), row=2, col=1)
    fig.add_trace(go.Scatter(x=np.arange(len(corr_array[0])), y=upper_y, mode='lines', line_color='rgba(255,255,255,0)',showlegend=False),row=2, col=1)
    fig.add_trace(go.Scatter(x=np.arange(len(corr_array[0])), y=lower_y, mode='lines',fillcolor='rgba(32, 146, 230,0.3)',fill='tonexty', line_color='rgba(255,255,255,0)',showlegend=False),row=2, col=1) 
    
    corr_array = pacf(series.dropna(),nlags=nlags, alpha=0.05) 
    lower_y = corr_array[1][:,0] - corr_array[0]
    upper_y = corr_array[1][:,1] - corr_array[0]
    
 
    [fig.add_trace(go.Scatter(x=(x,x), y=(0,corr_array[0][x]), mode='lines',line_color='#3f3f3f',showlegend=False), row=3, col=1) for x in range(len(corr_array[0]))]
    fig.add_trace(go.Scatter(x=[x for x in range(len(corr_array[0]))], y=corr_array[0],mode='markers', marker_color='#1f37b1',marker_size=8,name='PACF'  ), row=3, col=1)
    fig.add_trace(go.Scatter(x=np.arange(len(corr_array[0])), y=upper_y, mode='lines', line_color='rgba(255,255,255,0)',showlegend=False),row=3, col=1)
    fig.add_trace(go.Scatter(x=np.arange(len(corr_array[0])), y=lower_y, mode='lines',fillcolor='rgba(32, 146, 230,0.3)',fill='tonexty', line_color='rgba(255,255,255,0)',showlegend=False),row=3, col=1) 
    


    fig.update_layout(height = 500, width = 800) # the allows for zooming into 

    fig.add_hline(
    y = 0,
    line_width=1, line_color="black",
    row=2, col=1)

    fig.add_hline(
    y = 0,
    line_width=1, line_color="black",
    row=3, col=1)
    
    
    return [fig]


###############

'''
will set this later 
def find_order_of_differencing(ts):
    adf_res=ndiffs(ts, test='adf')
    kpss_res=ndiffs(ts, test='kpss')
    pp_res=ndiffs(ts, test='pp')
    
    print(f"Result from order of differencing tests: \n adf={adf_res}\n kpss={kpss_res}\n pp={pp_res}")
'''
