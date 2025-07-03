from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the machine learning model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        gender = request.form['gender']
        hemoglobin = float(request.form['hemoglobin'])
        mch = float(request.form['mch'])
        mchc = float(request.form['mchc'])
        mcv = float(request.form['mcv'])
        
        # Preprocess input data
        gender = 1 if gender == 'male' else 0  # Example: convert gender to numerical
        
        # Create a numpy array for prediction
        input_features = np.array([[gender, hemoglobin, mch, mchc, mcv]])
        
        # Make prediction
        prediction = model.predict(input_features)
        
        # Determine result based on prediction
        result = 'Anemic' if prediction[0] == 1 else 'Not Anemic'
        
        return render_template('result.html', result=result)
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
