from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load saved model and columns
model = joblib.load('model.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        tenure = float(request.form.get('tenure', 0))
        monthly_charges = float(request.form.get('MonthlyCharges', 0))
        total_charges = float(request.form.get('TotalCharges', 0))
        contract = request.form.get('Contract', 'Month-to-month')

        input_data = {
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'Contract': contract
        }

        df_input = pd.DataFrame([input_data])
        df_encoded = pd.get_dummies(df_input)
        df_final = df_encoded.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(df_final)[0]

        if prediction == 1:
            result = "Churn Alert: Customer is likely to leave!"
            status_class = "churn"
        else:
            result = "Customer is likely to stay."
            status_class = "no-churn"

        return render_template('index.html', prediction_text=result, status_class=status_class)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}", status_class="error")

if __name__ == '__main__':
    app.run(debug=True)
