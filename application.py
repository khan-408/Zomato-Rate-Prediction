from flask import Flask, request, render_template, jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictionPipeline



application = Flask(__name__)

app = application

# routing home page

@app.route('/')
def home_page():
    return render_template('index.html')

# routing prediction page

@app.route('/predict_datapoints',methods=['GET','POST'])
def predict_datapoints():
    if request.method=='GET':
        return render_template('index.html')
    else:
        # taking input as Custom Data
        data = CustomData(
            votes = float(request.form['Votes']),
            cost = float(request.form['Cost For 2 Person']),
            online_order = eval(request.form['Online Order']),
            book_table = eval(request.form['Book Table']),
            location = request.form['Location'],
            rest_type = request.form['Restaurant Type'],
            cuisines = request.form['Cuisines'],
            type = request.form['Type']
        )
        # Creating DataFrame of CustomData
        new_data = data.get_data_as_dataframe()
        #Initializing Predict pipeline
        predict_pipeline = PredictionPipeline()
        predicted = predict_pipeline.predict(new_data)
        #Asigning predcted value in result variable.
        results = round(predicted[0],2)

        return render_template('index.html', final_result ='Your Rating is: {}'.format(results))

if __name__=='__main__':
    application.run(host='0.0.0.0',debug=True)