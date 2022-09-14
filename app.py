from flask import Flask,render_template,request
import pickle
from flask_cors import cross_origin
import pandas as pd 
import numpy as np
import mysql.connector as conn
import sklearn



model = pickle.load(open('model1.pkl','rb'))
processing = pickle.load(open('transform2.pkl','rb'))
scale = pickle.load(open('scale.pkl','rb'))


app = Flask(__name__)

@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")




@app.route("/predict", methods = ["POST"])
@cross_origin()
def predict():
    if request.method == "POST":

        # Date_of_Journey
        date_dep = request.form["Dep_Time"]
        Journey_day = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").day)
        Journey_month = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").month)

        # Departure
        Dep_hour = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").hour)
        Dep_min = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").minute)
        # print("Departure : ",Dep_hour, Dep_min)

        # Arrival
        date_arr = request.form["Arrival_Time"]
        Arrival_hour = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").hour)
        Arrival_min = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").minute)
        Total_stops = int(request.form["stops"])
        year = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").year)
        airline=request.form['airline']
        destination = request.form["Destination"]
        Source = request.form["Source"]

        l = [airline,Source,destination,Total_stops,Journey_day,Journey_month,year,Arrival_hour,Arrival_min,Dep_hour,Dep_min]

        l1 = np.array(l)
        l2 = l1.reshape(1,-1)
        l3 = processing.transform(l2)
        l4 = scale.transform(l3)
        s = round(model.predict(l4)[0],2)







        return render_template('home.html',prediction_text="Your Flight price is Rs. {}".format(s))
    return render_template("home.html")  
if __name__ == '__main__':
    app.run(debug=True)