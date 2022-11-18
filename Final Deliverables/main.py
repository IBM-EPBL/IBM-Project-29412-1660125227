from flask import Flask,render_template,request
import os 
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import requests

app=Flask(__name__,template_folder="templates")
model=load_model('Nutrition-Analysis.h5')
print("Loaded model from disk")


@app.route('/home')
def home():
    return render_template("home.html")

@app.route('/image1',methods=['GET','POST'])
def image1():
    return render_template("image.html")

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        f=request.files['file']
        basepath=os.path.dirname('__file__')
        filepath=os.path.join(basepath,'static',f.filename)
        f.save(filepath)


        img=image.load_img(filepath,target_size=(64,64))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)

        pred=np.argmax(model.predict(x),axis=1)
        print("prediction",pred)

        index=["APPLES","BANANA","ORANGE","PINEAPPLE","WATERMELON"]

        result=str(index[pred[0]])
        x=result
        result=nutrition(result)

        return render_template("0.html",showcase=(result),showcase1=(f.filename))


def nutrition(index):

    url="https://calorieninjas.p.rapidapi.com/v1/nutrition"
    querystring={"query":index}

    headers={
            "X-RapidAPI-Key":"228bc54e2bmsh125425366c0edcdp11af24jsn5f87cef4e48e",
            "X-RapidAPI-Host": "calorieninjas.p.rapidapi.com"
            }

    response=requests.request("GET",url,headers=headers,params=querystring)
    print(response.json())
    return response.json()['items']











