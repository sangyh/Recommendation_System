
from flask import Flask,render_template,url_for, request, redirect,session
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from model import content_rec_engine
import json

app = Flask(__name__)
app.secret_key = "super secret key"
api = Api(app)

@app.route('/')
def home():
	return render_template('index.html')


# create new model object
model = content_rec_engine()

#load model
with open('recengine.pkl','rb') as f:
		model=pickle.load(f)


@app.route('/',methods=['POST'])
def parse_data():	
	if request.method == "POST":
		session['test_x']=request.form['inputmovie']
    
	#return 'Hey '+ processed_text

		# argument parsing
		parser = reqparse.RequestParser()
		parser.add_argument('query')
		# Setup the Api resource routing here
		# Route the URL to the resource
		return redirect ('/r')
		

@app.route('/r',methods=['GET','POST'])
def get_recs():	
	user_query = session.get('test_x').lower()
	recos = model.find_similar_movs(user_query)
   
    # create JSON object
	output = {'prediction': recos}
        
	return render_template('index2.html', output=output)


if __name__ == '__main__':
    app.run()







