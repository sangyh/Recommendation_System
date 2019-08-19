# Recommendation_System
Content and collaborative Recommendation systems

This repo shows the steps to build content and  collaborative recommendation systems using the Movielens dataset. 

The three types of recommendation systems implemented are indicated below.

Content-based : that uses only item similarity
Hybrid RS: that uses item similarity along with user ratings
Collaborative RS: only uses user ratings

The content-RS was also served as a Rest API using Flask Restful and deployed on Google App engine. The API endpoints is then accesssed using a Jupyter notebook called user_requests.

MovieLens data sets were collected by the GroupLens Research Project at the University of Minnesota.
This data set consists of:
* 100,000 ratings (1-5) from 943 users on 1682 movies. 
* Each user has rated at least 20 movies. 
