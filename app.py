from flask import Flask, request
from pickle import load
import numpy as np

app = Flask(__name__)

@app.route('/wine_model_prediction', methods=['GET'])
def wine_model_prediction():
	# Load the model
	model = load(open('model.pkl', 'rb'))
	# Load the scaler
	scaler = load(open('scaler.pkl', 'rb'))

	
	new_wine = np.array([request.args.get('fixed_acidity', type=float) ,
		   request.args.get('volatile_acidity', type=float) ,
		   request.args.get('citric_acid', type=float),
		   request.args.get('residual_sugar', type=float),
		   request.args.get('chlorides', type=float),
		   request.args.get('free_sulfur_dioxide', type=float),
		   request.args.get('total_sulfur_dioxide', type=float),
		   request.args.get('density', type=float),
		   request.args.get('pH', type=float),
		   request.args.get('sulphates', type=float),
		   request.args.get('alcohol', type=float)]).reshape(1, -1)
		   
	new_wine_scaled = scaler.transform(new_wine)
	prediction = model.predict(new_wine_scaled)
	return prediction.tolist()

if __name__ == '__main__':
	app.run(debug=True, port=5000)
