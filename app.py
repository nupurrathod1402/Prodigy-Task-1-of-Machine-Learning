from flask import Flask, request,render_template
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


app = Flask(__name__)

data_path = 'train.csv'
data = pd.read_csv(data_path)

data = data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'SalePrice']]

data['TotalBath'] = data['FullBath'] + 0.5 * data['HalfBath']


X = data[['GrLivArea', 'BedroomAbvGr', 'TotalBath']]
y = data['SalePrice']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


joblib.dump(model, 'house_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        gr_liv_area = float(request.form['GrLivArea'])
        bedrooms = float(request.form['BedroomAbvGr'])
        total_bath = float(request.form['TotalBath'])

        
        features = np.array([[gr_liv_area, bedrooms, total_bath]])

        
        scaled_features = scaler.transform(features)

        
        prediction = model.predict(scaled_features)

        # Convert the predicted price to rupees (assuming 1 USD = 83 INR)
        predicted_price_in_inr = round(prediction[0] * 86.20*0.5, 2)

        return render_template('result.html', prediction_text=f"Predicted Price: ₹{predicted_price_in_inr}")

    except Exception as e:
        return render_template('result.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, port=5001)