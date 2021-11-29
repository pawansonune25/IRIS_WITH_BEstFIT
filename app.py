from flask import Flask,render_template,request
import pickle
import numpy as np
app = Flask(__name__)
model = pickle.load(open("KNN_model.pkl","rb"))



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/PREDICT", methods = ["POST"] ) 
def pavan():
    SepalLengthCm =float(request.form.get("SepalLengthCm")) 
    SepalWidthCm =float(request.form.get("SepalWidthCm"))
    PetalLengthCm = float(request.form.get("PetalLengthCm"))
    PetalWidthCm =float(request.form.get("PetalWidthCm")) 

    result = model.predict(np.array([SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]).reshape(1,4))
    if result[0] == 0:
        result='Iris-setosa'
    elif result[0] == 1:
        result='Iris-virginica'
    else:
        result='Iris-versicolor'
    return render_template('index.html',prediction = result)
   


    return render_template("index.html",pavan = result)
if __name__ == "__main__":
    app.run(host= '0.0.0.0',port=8080,debug=True)

