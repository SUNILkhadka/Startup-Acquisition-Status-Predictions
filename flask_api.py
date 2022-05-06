from flask import Flask, jsonify,render_template,request
import pickle
import numpy as np
import pandas as pd
import xgboost

app = Flask(__name__,static_url_path='',template_folder='web/templates',static_folder='web/static')
model = pickle.load(open('finalized_model.pkl','rb'))

@app.route("/")
def home():
    return render_template('base.html')
    

@app.route("/predict",methods=['POST'])
def predict():
    features = np.array([x for x in request.form.values()])
    print(features)
    input_df = pd.DataFrame(features.reshape(1,9),columns=['funding_rounds', 'funding_total_usd','milestones', 
         'relationships', 'lat', 'lng','activeDays','category_code', 'country_code'])
    output = model.predict(input_df)
    print(output[0])
    if output[0] == 1:
        output = 'Closed'
    else:
        output = 'Not Closed'
    return render_template('base.html',prediction_test='The status of the company is : "{}"'.format(output))


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)