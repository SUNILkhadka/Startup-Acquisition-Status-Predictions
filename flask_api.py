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
    for i,x in enumerate(features):
        if x == '' or x=='nan' or x=='NaN':
            features[i] = np.nan
    input_df = pd.DataFrame(features.reshape(1,8),
                            columns=['founded_at','funding_rounds', 'funding_total_usd','milestones', 
                                    'relationships','activeDays','category_code', 'country_code']
                            )
    output = model.predict(input_df)
    prob = model.predict_proba(input_df)
    if output[0] == 1:
        output = 'Closed'
    else:
        output = 'Not Closed'
    return render_template('base.html',
                            prediction_test="The status of the company is : '{}'".format(output),
                            prediction_prob='[{:.2f}%,{:.2f}%]'.format(prob[0,0]*100,prob[0,1]*100)
                            )


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    input_data = pd.read_json(data)
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)
    return jsonify(result=int(prediction[0]),
                    prediction_prob=[float(prob[0,0]),float(prob[0,1])]
                    )
    # Another way
    # output = {'result':int(prediction[0]}
    # return jsonify(output)
    