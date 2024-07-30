import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

def fetch_binance_data(symbol, interval, limit):
    url = "https://api.binance.com/api/v3/klines"
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume',
                                     'close_time', 'quote_asset_volume', 'number_of_trades',
                                     'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['close'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    return df

def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def create_features(df):
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()
    df['RSI'] = compute_rsi(df['close'], 14)
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Bollinger_High'] = df['close'].rolling(window=20).mean() + 2 * df['close'].rolling(window=20).std()
    df['Bollinger_Low'] = df['close'].rolling(window=20).mean() - 2 * df['close'].rolling(window=20).std()
    df['volume_change'] = df['volume'].pct_change()
    
    # Lag features
    for i in range(1, 11):
        df[f'close_lag_{i}'] = df['close'].shift(i)
    
    df.dropna(inplace=True)  # Remove rows with NaN values

def create_lstm_data(df, original_close):
    X, y = [], []
    time_steps = 10
    for i in range(len(df) - time_steps - 7):  # Predict 7 days ahead
        X.append(df[i:(i + time_steps)].values)
        y.append(original_close[i + time_steps + 7])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(df):
    df['future_close'] = df['close'].shift(-7)  # Predict price 7 days ahead
    df.dropna(inplace=True)  # Remove rows with NaN values
    
    original_close = df['close'].values  # Save the original close values
    
    features = df[['SMA_50', 'SMA_200', 'RSI', 'EMA_12', 'EMA_26', 'MACD', 'MACD_signal', 'Bollinger_High', 'Bollinger_Low', 'volume_change'] + [f'close_lag_{i}' for i in range(1, 11)]]
    labels = df['future_close']
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    X, y = create_lstm_data(pd.DataFrame(scaled_features), original_close)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Mean Squared Error: {mse:.2f}")
    print(f"Model Accuracy (R² score): {r2:.2%}")
    
    return model, scaler

def predict_future_prices(model, scaler, df, conversion_rate):
    today = df['close_time'].max()
    next_week_start = today + timedelta(hours=1)  # Start from the next hour
    
    future_dates = pd.date_range(start=next_week_start, periods=168, freq='h')
    future_df = pd.DataFrame(index=future_dates)
    
    last_features = df[['SMA_50', 'SMA_200', 'RSI', 'EMA_12', 'EMA_26', 'MACD', 'MACD_signal', 'Bollinger_High', 'Bollinger_Low', 'volume_change'] + [f'close_lag_{i}' for i in range(1, 11)]].iloc[-10:]
    scaled_last_features = scaler.transform(last_features)
    
    future_features = np.array([scaled_last_features[-10:].values])
    predicted_closes = []
    
    for i in range(len(future_dates)):
        prediction = model.predict(future_features)
        predicted_closes.append(prediction[0, 0])
        new_feature = np.append(future_features[0, 1:], prediction[0, 0]).reshape((1, 10, -1))
        future_features = new_feature
    
    future_df['predicted_close'] = predicted_closes
    future_df['predicted_close_inr'] = future_df['predicted_close'] * conversion_rate
    
    return future_df

def get_conversion_rate():
    url = "https://api.exchangerate-api.com/v4/latest/USD"
    response = requests.get(url)
    data = response.json()
    conversion_rate = data['rates'].get('INR', 1)  # Default to 1 if INR rate is not found
    return conversion_rate

def main():
    symbol = input("Enter the cryptocurrency symbol (e.g., BTCUSDT): ")
    df = fetch_binance_data(symbol, '1h', 1000)
    create_features(df)

    model, scaler = train_model(df)
    if model is None:
        print("Failed to train the model. Exiting.")
        return

    conversion_rate = get_conversion_rate()
    future_df = predict_future_prices(model, scaler, df, conversion_rate)

    # Print the predicted prices in a tabular format
    print("\nPredicted Prices for {} in the next 168 hours:".format(symbol))

    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    
    print(future_df[['predicted_close', 'predicted_close_inr']].to_string(index=True, header=True, float_format=lambda x: '{:.8f}'.format(x)))

    pd.reset_option('display.max_rows')
    pd.reset_option('display.width')

    # Format the time to Hour:Minute
    formatted_times = future_df.index.strftime('%Y-%m-%d %H:%M')

    # Find the best buy and sell times based on the lowest and highest predicted prices
    best_buy_time = future_df.loc[future_df['predicted_close'].idxmin()].name.strftime('%Y-%m-%d %H:%M')
    best_sell_time = future_df.loc[future_df['predicted_close'].idxmax()].name.strftime('%Y-%m-%d %H:%M')

    print("\nBest Time to Buy for {}:".format(symbol))
    print("Time: {}, Price (USD): {:.8f}, Price (INR): {:.8f}".format(best_buy_time, future_df['predicted_close'].min(), future_df['predicted_close_inr'].min()))

    print("Best Time to Sell for {}: ".format(symbol))
    print("Time: {}, Price (USD): {:.8f}, Price (INR): {:.8f}".format(best_sell_time, future_df['predicted_close'].max(), future_df['predicted_close_inr'].max()))

if __name__ == "__main__":
    main()
