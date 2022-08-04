from flask import Flask,render_template,Response,request
from flask_cors import CORS
import flask
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pymongo
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
db = myclient.database_chatbot
collection = db.questions
r = collection.find({ })
list_ofintents=r[1:]
train=[]
target=[]
for q in list_ofintents:
    for p in q["patterns"]:
        train.append(p)
        target.append(q["tag"])
model = make_pipeline(TfidfVectorizer(analyzer='word',stop_words= 'english'), MultinomialNB())
model.fit(train,target)
app = Flask(__name__)
CORS(app)
@app.route("/",methods =["GET"])
def hello():
    return render_template('sample.html')
@app.route("/get")
def res():
    question=request.args.get('msg') 
    li=predict_class(question)
    res = get_response(li,question)
    print(len(res))
    return json.dumps(res)
def predict_class(ques):
    return model.predict([ques])
def get_response(ints,intents_json):
    if len(ints)==0:
        return "Sorry don't understand"
    tag = ints[0]
    for i in list_ofintents:
        if i["tag"] == tag:
            result = i["responses"]
            return result
if __name__ == "__main__":
    app.run("localhost", 6969,debug=True)