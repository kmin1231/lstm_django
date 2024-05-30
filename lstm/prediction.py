import io, base64
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from .models import SS, load_ss
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential,load_model
from keras.layers import LSTM, Dense, Dropout, Flatten
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

predict_start = '2024-05-01'

def getprediction():
  model2 = load_model('C:\\Users\\user\\Desktop\\lstm0529\\lstm\\lstm_model.h5')

  time_step = 10
  epochs = 5
  batch_size = 8
  prediction_results = []

  # predict_start = '2024-05-01'
  end_date = datetime.today().strftime('%Y-%m-%d')

  current_date = datetime.strptime(predict_start, '%Y-%m-%d')

  while current_date <= datetime.strptime(end_date, '%Y-%m-%d'):
    ahead = current_date - timedelta(days=30)
    try:
      df_ss = fdr.DataReader('005930', ahead, current_date)
      df_ks = fdr.DataReader('KS11', ahead, current_date)

      df_ss_common = df_ss[df_ss.index.isin(df_ks.index)]
      df_ks_common = df_ks[df_ks.index.isin(df_ss.index)]

      df_ss_common.dropna(inplace=True)
      df_ks_common.dropna(inplace=True)

      if df_ss_common.empty:
        print(f"No common dates found for {current_date}")
        current_date += timedelta(days=1)
        continue

      if current_date in df_ss_common.index:
        ss_close = df_ss_common.loc[current_date, 'Close']
        ks_close = df_ks_common.loc[current_date, 'Close']

        ss_close = df_ss_common['Close'].values.reshape(-1, 1)
        ks_close = df_ks_common['Close'].values.reshape(-1, 1)

        scaler_ss = MinMaxScaler()
        scaler_ks = MinMaxScaler()

        scaled_ss = scaler_ss.fit_transform(ss_close.reshape(-1, 1))
        scaled_ks = scaler_ks.fit_transform(ks_close.reshape(-1, 1))

        combined_data = []
        for i in range(len(scaled_ss) - time_step):
          combined_data.append(np.concatenate((scaled_ss[i:i+time_step], scaled_ks[i:i+time_step]), axis=1))
        combined_data = np.array(combined_data)
        X_train, X_test, y_train, y_test = train_test_split(combined_data[:, :-1], combined_data[:, -1], test_size=0.1, random_state=42)

        model2.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        predicted_data_scaled = model2.predict(X_test)
        predicted_data = scaler_ss.inverse_transform(predicted_data_scaled)
        prediction = predicted_data[-1]

        prediction_results.append((current_date, prediction))
        print(f'Prediction for {current_date.strftime("%Y.%m.%d")} : {prediction}')

    except ValueError as e:
        print(f"Skipping {current_date}: {e}")
        pass

    current_date += timedelta(days=1)
  return prediction_results


def predict_graph():
    prediction_results = getprediction()    

    today = datetime.today().strftime('%Y-%m-%d')
    # predict_start = datetime.today() - timedelta(days=60)

    df_prediction = pd.DataFrame(prediction_results, columns=['Date', 'Prediction'])
    df_prediction['Date'] = pd.to_datetime(df_prediction['Date'])
    df_prediction.set_index('Date', inplace=True)

    ss_prices = SS.objects.filter(date__range=(predict_start, today)).order_by('date')
    dates = [price.date for price in ss_prices]
    close = [price.close for price in ss_prices]

    df_actual = pd.DataFrame({'Date': dates, 'Close': close})
    df_actual.set_index('Date', inplace=True)

    plt.figure(figsize=(7, 4))
    plt.plot(df_actual.index, df_actual['Close'], label='Actual')
    plt.plot(df_prediction.index, df_prediction['Prediction'], label='Prediction')
    plt.title('Actual vs. Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=30)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    plt.close()
    img_data.seek(0)
    graph_img = base64.b64encode(img_data.read()).decode('utf-8')

    return graph_img