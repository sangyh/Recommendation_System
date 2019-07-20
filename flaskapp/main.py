
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from model import content_rec_engine


app = Flask(__name__)
api = Api(app)

# create new model object
model = content_rec_engine()

#load model
with open('recengine.pkl','rb') as f:
		model=pickle.load(f)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')

class get_recs(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']
        print("this is the user query ",user_query)
        
        recos = model.find_similar_movs(user_query)
   
        # create JSON object
        output = {'prediction': recos}
        
        return output

# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(get_recs, '/')

if __name__ == '__main__':
    app.run(debug=True)







