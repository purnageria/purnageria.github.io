#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import requests
from prophet import Prophet
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup as soup
#model = pickle.load(open("Stock_Prediction.pkl", "rb")

# In[7]:


app = Flask(__name__)


# In[3]:


@app.route('/')
def home():
    return render_template("home.html")


# In[4]:

 


@app.route('/get_data', methods=['POST'])

def get_data():
    stock = request.form["stock"]
    #model = pickle.load(open("Stock_Prediction.pkl", "rb"))
    url = "https://finance.yahoo.com/quote/"+stock+"/history?p="+stock

    req = Request(url , headers={'User-Agent': 'Mozilla/5.0'})

    webpage = urlopen(req).read()
    page_soup = soup(webpage, "html.parser")
    extract_link = page_soup.find("div", class_ = "YDC-Col1 Bdendc(t) Bdendw(340px) tablet_Bdendw(0)--noRightRail Bdends(s) Mt(17px) Pos(r) Z(1)").find("div", class_ = "C($tertiaryColor) Mt(20px) Mb(15px)").find("a")
    link = extract_link.get("href")

    data = pd.read_csv(link)
    html = data.to_html()
    
    model = Prophet()
    
    columns = ["Date", "Close"]

    new_df = pd.DataFrame(data, columns=columns)

    prophet_df = new_df.rename(columns= {"Date":"ds", "Close":"y"})
    model.fit(prophet_df)
    
    days = request.form["days"]
    
    next = model.make_future_dataframe(periods=int(days))
    forecast = model.predict(next)
    fore = forecast.tail(int(days))
    
    fore.rename(columns= {"ds":"Date","yhat":"Prediction", "yhat_lower":"Low","yhat_upper":"High"}, inplace = True)
    fore.drop("multiplicative_terms", axis=1, inplace =True)
    fore.drop("multiplicative_terms_lower", axis=1, inplace =True)
    fore.drop("multiplicative_terms_upper", axis=1, inplace =True)
    fore.drop("additive_terms", axis=1, inplace =True)
    fore.drop("additive_terms_lower", axis=1, inplace =True)
    fore.drop("additive_terms_upper", axis=1, inplace =True)
    fore.drop("weekly", axis=1, inplace =True)
    fore.drop("weekly_lower", axis=1, inplace =True)
    fore.drop("weekly_upper", axis=1, inplace =True)
    fore.drop("trend_lower", axis=1, inplace =True)
    fore.drop("trend_upper", axis=1, inplace =True)
    fore['Date'] = fore['Date'].dt.date
#     fore.drop("ds", axis =1, inplace = True)
    html_fore = fore.style.set_properties(**{'background-color': 'black',                                                   
                                    'color': 'lawngreen',                       
                                    'border-color': 'white'}).hide_index().to_html()
    
    return html_fore
  
   
               
                 


# In[5]:


#@app.route('/result', methods=['POST'])
#def result():
   #loaded_model = pickle.load(open("Stock_data.pkl", "rb"))
   # s = request.form.to_dict()
   # to_predict_list= list(s.values())
   # result = loaded_model(to_predict_list[0])
    #return flask.render_template('result.html', prediction = result)


# In[6]:


# Main function
if __name__ == "__main__":
    app.run(debug=True)
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    


# In[ ]:




