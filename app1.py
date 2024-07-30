import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
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

def create_feature_label_data(df):
    df['future_close'] = df['close'].shift(-7)  # Predict price 7 days ahead
    df.dropna(inplace=True)  # Remove rows with NaN values

    features = df[['SMA_50', 'SMA_200', 'RSI', 'EMA_12', 'EMA_26', 'MACD', 'MACD_signal', 'Bollinger_High', 'Bollinger_Low', 'volume_change'] + [f'close_lag_{i}' for i in range(1, 11)]]
    labels = df['future_close']
    return features, labels

def build_and_train_model(features, labels):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Mean Squared Error: {mse:.2f}")
    print(f"Model Accuracy (R² score): {r2:.2%}")
    
    return model, scaler

def predict_future_prices(model, scaler, df, conversion_rate):
    today = df['close_time'].max()
    next_hour_start = today + timedelta(hours=1)  # Start from the next hour

    future_dates = pd.date_range(start=next_hour_start, periods=168, freq='h')
    future_df = pd.DataFrame(index=future_dates)
    
    # Use the last available features for the prediction
    features_count = len(df[['SMA_50', 'SMA_200', 'RSI', 'EMA_12', 'EMA_26', 'MACD', 'MACD_signal', 'Bollinger_High', 'Bollinger_Low', 'volume_change'] + [f'close_lag_{i}' for i in range(1, 11)]].columns)
    
    last_features = df[['SMA_50', 'SMA_200', 'RSI', 'EMA_12', 'EMA_26', 'MACD', 'MACD_signal', 'Bollinger_High', 'Bollinger_Low', 'volume_change'] + [f'close_lag_{i}' for i in range(1, 11)]].iloc[-1].values
    last_features = np.reshape(last_features, (1, features_count))  # Reshape for prediction
    
    future_predictions = []
    
    for _ in range(168):  # Predict for the next 168 hours
        # Predict the price for the next hour
        scaled_last_features = scaler.transform(last_features)
        prediction = model.predict(scaled_last_features)
        future_predictions.append(prediction[0])
        
        # Update last_features for next prediction
        new_feature = np.zeros((1, features_count))
        last_features = np.roll(last_features, shift=-1, axis=1)  # Roll features left
        last_features[0, -1] = prediction  # Update last feature with the new prediction

    future_df['predicted_close'] = future_predictions
    future_df['predicted_close_inr'] = future_df['predicted_close'] * conversion_rate

    # Find best times to buy and sell
    min_index = future_df['predicted_close'].idxmin()
    max_index = future_df['predicted_close'].idxmax()

    best_time_to_buy = future_df.loc[min_index].name
    best_time_to_sell = future_df.loc[max_index].name
    best_price_to_buy = future_df.loc[min_index, 'predicted_close']
    best_price_to_sell = future_df.loc[max_index, 'predicted_close']
    best_price_to_buy_inr = best_price_to_buy * conversion_rate
    best_price_to_sell_inr = best_price_to_sell * conversion_rate

    print(f"\nBest time to buy: {best_time_to_buy} at price ${best_price_to_buy:.8f} ({best_price_to_buy_inr:.2f} INR)")
    print(f"Best time to sell: {best_time_to_sell} at price ${best_price_to_sell:.8f} ({best_price_to_sell_inr:.2f} INR)")

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

    features, labels = create_feature_label_data(df)
    model, scaler = build_and_train_model(features, labels)

    conversion_rate = get_conversion_rate()
    future_df = predict_future_prices(model, scaler, df, conversion_rate)

    # Print the predicted prices in a tabular format
    print("\nPredicted Prices for {} in the next 168 hours:".format(symbol))

    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    
    print(future_df[['predicted_close', 'predicted_close_inr']].to_string(index=True, header=True, float_format=lambda x: '{:.8f}'.format(x)))

    pd.reset_option('display.max_rows')
    pd.reset_option('display.width')

if __name__ == "__main__":
    main()
