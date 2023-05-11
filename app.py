import pandas as pd
import numpy as np 
import pickle 
from flask import Flask,render_template,request



app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def home():
    sheet_id="1h8JSMzs8NLrB9FCkOwpH1JNyvZ5VgVML4363jmJRPIY" 
    df=pd.read_csv(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv")

    model=pickle.load(open("model.pkl",'rb'))
    print(df.columns)

    if(len(df['temperature'])==0 or len(df['humidity'])==0 or len(df['moisture'])==0):
        print("NO data in the sheet")
    else:
        print(df)
        temperature=np.mean(df['temperature'])
        humidity=np.mean(df['humidity'])
        moisture=np.mean(df['moisture'])
        print([temperature,humidity,moisture])
        crop=model.predict([[temperature,humidity,moisture]])
        print("The predicted crop is:",crop)
    return render_template('basic.html',crop=crop)

if(__name__=="__main__"):
    app.run(debug=True)