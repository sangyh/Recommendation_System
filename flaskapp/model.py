import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class content_rec_engine(object):

	def __init__(self):
		model=[]

	def load_data(self,path):
		self.u_info = pd.read_csv(path+'/u.info') #, header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])

		self.u_genre = pd.read_csv(path+'/u.genre',sep='|')
		self.u_genre.drop(labels='0',axis=1,inplace=True)
		self.u_genre.rename(index=str, columns={"unknown": "genre"},inplace=True)

		i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure', 
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
		self.items = pd.read_csv(path+'/u.item', sep='|', names=i_cols, encoding='latin-1')
		print('Items dataframe shape: ',self.items.shape)
		self.items.set_index('movie id', inplace=True)

		self.users = pd.read_csv(path+'/u.user', sep='|', names=['user id','age', 'gender', 'occupation','zip code'],encoding='latin-1')
		print('Users dataframe shape: ',self.users.shape)
		self.users.set_index('user id',inplace=True)

		self.ratings = pd.read_csv(path+'/u.data', sep='\t', names=['user_id','movie id', 'rating', 'timestamp'],encoding='latin-1')
		self.ratings.drop(labels='timestamp',axis=1,inplace=True)

	def create_feature_lists(self):
		self.item_features=self.items.drop(labels=['movie title','release date','video release date','IMDb URL','unknown'],axis=1)
		self.item_features.drop(self.item_features.index[266],inplace=True)
		self.item_features.drop(self.item_features.index[1371],inplace=True)
		#print(' normalization of feature vectors..')
		#row_sum=self.item_features.sum(axis=1)
		#self.item_features=self.item_features.divide(np.sqrt(np.array(row_sum)),axis=0)

		#create IDF vectors
		column_sum=self.item_features.sum(axis=0)
		self.IDF=list(np.log10(len(self.item_features)/column_sum))

		#calculate weighted scores for each attribute using IDF
		self.item_features=self.item_features.multiply(self.IDF,axis=1)

	def find_similar_movs(self,mov):
		movie_id_liked=self.items[self.items['movie title'].str.contains(mov)].index.values
		target_features=self.item_features.loc[movie_id_liked[0]]

		#dot product to find similarities with all movies
		target_features=target_features.values.reshape(1, -1)
		cosine_sim=cosine_similarity(self.item_features,target_features)
		cosine_sim=cosine_sim.flatten()
		results_df=pd.DataFrame(cosine_sim,list(self.item_features.index.values))

		suggested=results_df.sort_values(by=0,ascending=False).index.values
		print(suggested[:10])
		top10recs=[self.items.loc[i]['movie title'] for i in suggested[:10]]

		print(top10recs) 

		return top10recs




	def pickle_rec(self, path='recengine.pkl'):
		with open(path,'wb') as f:
			pickle.dump(self,f)
			print('Pickled rec engine at {}'.format(path))





