import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class content_rec_engine(object):

	def __init__(self):
		model=[]

	def load_data(self,path):
		
		cols = ['movie id', 'movie title' , '(no genres listed)', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western','IMAX']

		genre_names = [ '(no genres listed)', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western','IMAX']

		self.movies_df=pd.DataFrame()

		with open(path+'/movies.dat','r', encoding="utf8") as f:
		    for l in f:
		        arr=[0]*len(genre_names)
		        genres=l.rstrip('\n').split('::')[2].split('|')
		        i=l.split('::')[0]
		        title=l.split('::')[1]
		        for g in genres:
		            arr[genre_names.index(g)]=1
		        
		        #print(arr)
		        arr.insert(0,title)
		        arr.insert(0,i)
		        self.movies_df=self.movies_df.append(pd.Series(arr),ignore_index=True)


		col_map=self.col_mapper(cols)
		self.movies_df=self.movies_df.rename(columns=col_map)

		self.movies_df.drop(self.movies_df.index[self.movies_df['(no genres listed)']==1],axis=0,inplace=True)

		#write movie titles into csv for html dropdpwn
		self.movies_df.to_csv('export_movie_titles.csv',columns=['movie title'],header=False, index=False)

	def col_mapper(self,cols):
		    return dict((cols.index(v), v) for v in cols)		

	def create_feature_lists(self):
		self.item_features=self.movies_df.drop(labels=['movie id','movie title','(no genres listed)'],axis=1)
		

		#create IDF vectors
		column_sum=self.item_features.sum(axis=0)
		self.IDF=list(np.log10(len(self.item_features)/column_sum))

		#calculate weighted scores for each attribute using IDF
		self.item_features=self.item_features.multiply(self.IDF,axis=1)

	def find_similar_movs(self,mov):
		movie_id_liked=self.movies_df[self.movies_df['movie title'].str.lower().str.contains(mov)].index.values
		if len(movie_id_liked)==0:
		    return('movie not found. please try again..')
		target_features=self.item_features.loc[movie_id_liked[0]]

		#dot product to find similarities with all movies
		target_features=target_features.values.reshape(1, -1)
		cosine_sim=cosine_similarity(self.item_features,target_features)
		cosine_sim=cosine_sim.flatten()
		results_df=pd.DataFrame(cosine_sim,list(self.item_features.index.values))

		suggested=results_df.sort_values(by=0,ascending=False).index.values
		print(suggested[:10])
		top10recs=[self.movies_df.loc[i] for i in suggested[:10]]

		results=[]
		for i in top10recs:
		    content=[]
		    content.append(i['movie title'])
		    
		    for j in i.index[2:]:    
		        if i[j]>0:
		            #print(j,i[j])
		            content.append(j)
		    results.append(content)

		return results



	def pickle_rec(self, path='recengine.pkl'):
		with open(path,'wb') as f:
			pickle.dump(self,f)
			print('Pickled rec engine at {}'.format(path))





