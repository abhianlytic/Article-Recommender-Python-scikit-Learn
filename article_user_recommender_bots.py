
# coding: utf-8

# # IMPORT all the relevant packages

# In[59]:

import pip

def install(package):
    pip.main(['install', package])

# Example
if __name__ == '__main__':
    install('argh')
    install('pandas')
    install('nltk')
    install('re')
    install('gensim')
    install('numpy')
    install('os')
    install('scikit-learn')
    install('pyLDAvis')
    install('time')
    install('matplotlib')

#NLTK Packages
import nltk
nltk.download('all')

from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()

from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()

import re, gensim, os
import pandas as pd
import numpy as np
import math
#import spacy

#SCIKIT LEARN Packages
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF, LatentDirichletAllocation

from sklearn.model_selection import GridSearchCV
import pyLDAvis.sklearn


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors


from time import time

import matplotlib.pyplot as plt
#%matplotlib inline


# # Read Article Data

# In[60]:

#Read the article data file from the saved location
file_location =input("Please enter the Article Data Location(e.g - G:/x/y) : ")
# file_location = G:/analytics/ISB CBA/Residency/Residency3&4/B9MT2DMG1/DMG 1 Assignment and Dataset-20180207
# file_name = news_articles
file_name = input("Please enter the file name(without extension) : ")
click_name = input("Please enter the clickstream file name(clickstream)")
article_data = pd.read_csv(file_location+"/"+file_name+".csv")
clickstream = pd.read_csv(file_location+"/"+click_name+".csv")
#list(article_data.columns.values)#columns in the article data


# # Read Stopword File

# In[61]:

#Now we will read stopwords file, which we have created after several 
#iterations.

os.chdir(file_location)
def stopwords_list(filename):
    stopwords = set()
    with open(filename, "r") as file:
        for line in file:
            word = line.strip()
            stopwords.add(word.lower())
        return(stopwords)

stop_words = stopwords_list("stopwords.txt")
print ("There are total "+str(len(stop_words))+ " Stopwords extracted after several iterations on news Articles.")


# # Pre-Processing of Article Contents

# In[62]:

#Pre-processing Data : 1 - lower case conversion of content

temp = []
for i in article_data['Content']:
    i = str(i)
    contents = i.split()
    cleancontents = [x.lower() for x in contents]
    cleancontents1 = " ".join(cleancontents)
    temp.append(cleancontents1)

#create new column in the article data to store the lower case contents.
article_data['CleanContents'] = temp


# In[63]:

#Pre-processing Data : 2 - Remove stop words

temp = []
for i in article_data['CleanContents']:
    content_tokens = word_tokenize(i)
    nostopwords = [x for x in content_tokens if x not in stop_words]
    contentnostopwords = " ".join(nostopwords)
    temp.append(contentnostopwords)

#update the CleanContents column
article_data['CleanContents'] = temp


# In[64]:

#Pre-processing Data : 3 -Stem/lemmatize the words and then remove duplicates

def unique(w):
    ulist = []
    [ulist.append(x) for x in w if x not in ulist]
    return ulist

temp = []
for i in article_data['CleanContents']:
    c = word_tokenize(i)
    m= []
    for j in c:
        #stemword = stem.stem(j)
        lemword = lem.lemmatize(j)
        #m.append(stemword)
        m.append(lemword)
    temp1 = " ".join(m)
    temp.append(temp1)

temp2 = []
for i in temp:
    temp22 = " ".join(unique(i.split()))
    temp2.append(temp22)

article_data['Ccontent'] = temp2
#article_data.head()


# # Split the data into train and test
# # For the given scenario of unsupervised learning , train contains all the data, this is just for flow.

# In[65]:

#Pre-processing completed, now we will split the data into train and test.

train, test = train_test_split(article_data, test_size = 0)


# In[66]:

#create list of corpus
sample_train = []
for i in train['Ccontent']:
    sample_train.append(i)
#sample_train[0:2]


# In[67]:

#further preprocessing
sample_train = [re.sub('\S*@\S*\s?','',i) for i in sample_train]
sample_train = [re.sub('\s+',' ',i) for i in sample_train]
sample_train = [re.sub("\'","",i) for i in sample_train]
#sample_train[0:2]


# # Representation of Words in Machine Understable Language

# # 1. Create Document Term Matrix

# In[68]:

#1- Term document Matrix#initiating CountVectorizer(with default parameters) to convert text into a matrix of token counts.



countvect = CountVectorizer(stop_words=stop_words,max_df=.95,min_df=5)
#,token_pattern='[a-zA-Z0-9]{3,}'
countvect


# In[69]:

#Learn the vocabulary of the training data
t0= time()
countfit=countvect.fit(sample_train)
print("Completed in %0.4fs." % (time() - t0))


# In[70]:

#check the fitted vocabulary
countfeature = countvect.get_feature_names()
#countfeature[0:20]


# In[71]:

#convert the vocab into document term matrix(sparse matrix)
dtm = countvect.transform(sample_train)
dtm
#sparse matrix only contain values with index- thus require less space than other form of matrix


# In[72]:

#convert sparse matrix to dense matrix
dtm_array = dtm.toarray()
#dtm_array


# In[73]:

#document term matrix and vocabulary together
dtm_df = pd.DataFrame(dtm_array,columns=countfeature)
dtm_df.head()


# In[74]:

print (" Count vector Sparsicity: ", ((dtm_array > 0 ).sum()/dtm_array.size)*100, "%")


# # Create TFIDF

# In[75]:




tfidfvect = TfidfVectorizer(max_df=.95,min_df=2,stop_words=stop_words, use_idf = True,token_pattern='[a-zA-Z0-9]{3,}')
t0=time()
tfidffit = tfidfvect.fit_transform(sample_train)
print("Completed in %0.4fs." % (time() - t0),tfidfvect)


# In[76]:

tfidffeature = tfidfvect.get_feature_names()
#tfidffeature[0:10]


# In[77]:

#tfidf_tr = tfidfvect.transform(sample_train)


# In[78]:

tfidf_array = tfidffit.toarray()
#tfidf_array 


# In[79]:

tfidf_df = pd.DataFrame(tfidf_array, columns = tfidffeature)
tfidf_df.head()


# In[80]:

tfidf_df.shape


# In[81]:

print (" TFIDF vector Sparsicity: ", ((tfidf_array > 0 ).sum()/tfidf_array.size)*100, "%")


# # 3. Create LSA - Latent Semantic Analysis
# fittfidf = UVW
# U(doc,concepts)m*k 
# V(Diagonal Matrix)k*k
# W(Transpose Matrix)k*m
# 
# # Attributes:
# components_ : array, shape (n_components, n_features)
# explained_variance_ : array, shape (n_components,)
# The variance of the training samples transformed by a projection to each component.
# 
# explained_variance_ratio_ : array, shape (n_components,)
# Percentage of variance explained by each of the selected components.
# 
# singular_values_ : array, shape (n_components,)
# 
# The singular values corresponding to each of the selected components. The singular values are equal to the 2-norms of the n_components variables in the lower-dimensional space.

# # X = USV(T)
# # U - m(doc)*k(concepts)
# # S - k * k variance captured by each concepts, basically a diagonal matrix
# # V - m(terms)*k(concepts)

# from sklearn.decomposition import TruncatedSVD
# 
# lsavect = TruncatedSVD(n_iter=10,n_components=10,random_state=123)
# lsafit = lsavect.fit(tfidffit)
# print("Completed in %0.4fs." % (time() - t0),lsafit)

# #represent rows of V
# lsavect.components_

# lsavect.explained_variance_

# lsavect.explained_variance_ratio_

# lsavect.explained_variance_ratio_.sum()

# lsavect.singular_values_

# lsafeatures = tfidfvect.get_feature_names()
# lsafeatures[0:10]

# for i, comp in enumerate(lsavect.components_):
#     featurescomp = zip(lsafeatures,comp)
#     sortfeatures = sorted(featurescomp,key = lambda x : x[1], reverse=True)[:10]
#     print ("Concept %d:" %i)
#     for feature in sortfeatures:
#         print (feature)
#     print (" ")

# # 4. Create LDA - Latent Dirichlet Allocation

# In[82]:


ldavect = LatentDirichletAllocation(n_components=10,max_iter=50,learning_method ='online',learning_offset=50.,random_state=123)
t0=time()
ldafit = ldavect.fit(dtm)
print("Completed in %0.4fs." % (time() - t0), ldafit)


# In[83]:

ldavect.components_[0][0:10]


# In[84]:

ldafeatures = countvect.get_feature_names()
#ldafeatures[0:20]


# In[85]:

for i, comp in enumerate(ldavect.components_):
    featurescomp = zip(ldafeatures,comp)
    sortfeatures = sorted(featurescomp,key = lambda x : x[1], reverse=True)[:15]
    print ("Concept %d:" %i)
    for feature in sortfeatures:
        print (feature)
    print (" ")


# In[86]:

print ("Log Liklihood", ldavect.score(dtm)) # higher the better


# In[87]:

# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Perplexity: ", ldavect.perplexity(dtm))


# In[88]:

# See model parameters
print(ldavect.get_params())


# In[89]:

#create a transform dataframe for the LDA model
ldadf_tr = ldavect.fit_transform(dtm)
ldadf_tr
print("Completed in %0.4fs." % (time() - t0))


# In[90]:

#Now we will optimize using grid search and find the best parameters to model the topic using LDA
grid_params = {'n_components': [5,8,10,15], 'learning_decay': [.5, .7, .9]}


# In[91]:

lda = LatentDirichletAllocation()


# In[92]:


model = GridSearchCV(lda, param_grid=grid_params)


# In[93]:

t0=time()
model.fit(dtm)
print("Completed in %0.4fs." % (time() - t0))


# In[94]:

# Best Model
ldabestmodel = model.best_estimator_


# In[95]:

# Model Parameters
print("Best Model's Parameters: ", model.best_params_)


# In[96]:

# Log Likelihood Score
print("Best Model's Log Likelihood Score: ", model.best_score_)


# In[97]:

# Perplexity
print("Model Perplexity: ", ldabestmodel.perplexity(dtm))


# In[98]:

# Get Log Likelyhoods from Grid Search Output

n_components = [5,8,10,15]

loglikelyhoods5 = [round(gscore.mean_validation_score) for gscore in model.grid_scores_ if gscore.parameters['learning_decay']==0.5]
loglikelyhoods7 = [round(gscore.mean_validation_score) for gscore in model.grid_scores_ if gscore.parameters['learning_decay']==0.7]
loglikelyhoods9 = [round(gscore.mean_validation_score) for gscore in model.grid_scores_ if gscore.parameters['learning_decay']==0.9]

# Show graph
plt.figure(figsize=(15, 10))
plt.plot(n_components, loglikelyhoods5, label='0.5')
plt.plot(n_components, loglikelyhoods7, label='0.7')
plt.plot(n_components, loglikelyhoods9, label='0.9')
plt.title("Selecting Optimal LDA Model")
plt.xlabel("Number of Topics")
plt.ylabel("Log Likelyhood Scores")
plt.legend(title='Learning decay', loc='best')
plt.show()


# In[99]:

# create Doc-Topic Matrix
best_lda =ldabestmodel.transform(dtm)
#best_lda


# In[100]:

# Create Pandas Dataframe
topicnames = ["Topic" + str(i) for i in range(ldabestmodel.n_components)]
Article_Id = [str(i) for i in range(len(sample_train))]
print (topicnames)


# In[101]:

# Create Document - Topic Matrix

document_topic = pd.DataFrame(np.round(best_lda, 2), columns=topicnames, index=Article_Id)
document_topic.index.name = 'Article_Id'

relevant_topic = np.argmax(document_topic.values, axis=1)
document_topic['topic_pattern'] = relevant_topic

# Styling
def color_blue(val):
    color = 'blue' if val > .1 else 'red'
    return 'color: {col}'.format(col=color)

def boldfont(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)

# Apply Style
document_topics = document_topic.head(20).style.applymap(color_blue).applymap(boldfont)
document_topics


# In[102]:

topic_distribution = document_topic['topic_pattern'].value_counts().reset_index(name="Num of Documents")
topic_distribution.columns = ['Topic pattern', 'Num of Documents']
topic_distribution


# In[103]:


#pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(ldabestmodel, dtm, countvect, mds='tsne')
panel


# In[104]:

# Topic-feature Matrix
topic_features = pd.DataFrame(ldabestmodel.components_)


topic_features.columns = countvect.get_feature_names()
topic_features.index = topicnames


topic_features.head()


# In[105]:

# Show top n keywords for each topic
def show_topics(vectorizer=countvect, lda_model=ldabestmodel, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

topic_keywords = show_topics(vectorizer=countvect, lda_model=ldabestmodel, n_words=15)        

# Topic - Keywords Dataframe
topic_words = pd.DataFrame(topic_keywords)
topic_words.columns = ['Word'+str(i) for i in range(topic_words.shape[1])]
topic_words.index = ['Topic'+str(i) for i in range(topic_words.shape[0])]
topic_words


# # k-means clustering

# In[106]:

# k-means clustering

clusters = KMeans(n_clusters=5, random_state=100).fit_predict(best_lda)

# Singular Value Decomposition(SVD) model
svd_model = TruncatedSVD(n_components=2)  # 2 components
lda_svd = svd_model.fit_transform(best_lda)


x = lda_svd[:, 0]
y = lda_svd[:, 1]

# Weights for the 5 columns of best_lda, for each component
print("Component's weights: \n", np.round(svd_model.components_, 2))

# Percentage of total information in 'best_lda' explained by the two components
print("Variance Explained % : \n", np.round(svd_model.explained_variance_ratio_, 2))


# In[107]:

# Plot
plt.figure(figsize=(12, 12))
plt.scatter(x, y, c=clusters)
plt.xlabel('Component 2')
plt.xlabel('Component 1')
plt.title("Topic Clusters Partition", )


# In[108]:

article_pattern = document_topic[['topic_pattern']]
article_pattern = article_pattern.reset_index()
article_pattern['Article_Id'] = article_pattern['Article_Id'].apply(pd.to_numeric)
article_pattern.dtypes


# In[109]:

#Read the article data file from the saved location

#article_data = pd.read_csv("G:/analytics/ISB CBA/Residency/Residency3&4/B9MT2DMG1/DMG 1 Assignment and Dataset-20180207/news_articles.csv")
list(article_data.columns.values)#columns in the article data


# In[110]:

article_data.dtypes


# In[111]:

article_data.head()


# In[112]:


article_data_pattern = pd.merge(article_pattern,article_data,on='Article_Id')
article_data_pattern.head()


# # Need to build a condition for identifying new user

# In[113]:

def new_user_recommendation():
    new_user_target = article_data_pattern.sample(n=10)
    #dt = as.DataFrame(document_topics)
    return new_user_target['Article_Id']


# # User Profiling

# In[114]:


#Average speed of reading in word per minute
wpm = 180


# In[115]:

#read the clickstream data
#clickstream = pd.read_csv("G://analytics//ISB CBA//Residency//Residency3&4//B9MT2DMG1//DMG 1 Assignment and Dataset-20180207//clickstreamr.csv")
#clickstream.head()


# In[116]:

#Create user profile by aelecting few relevant columns from the clickstream data
user_profile = []
user_profile=clickstream[['Name','User ID','Article_Id','Time Spent']]
user_profile.head()


# In[117]:

#Create Article profile by aelecting few relevant columns from the Article data
article_profile = []
article_profile = article_data[['Article_Id','Content']]
article_profile.head()


# In[118]:

#Now merge the user profile and Article profile to get the view of what portion of data read by any user of any article
user_article = pd.merge(user_profile,article_profile,on='Article_Id')
user_article['Total_Words'] = user_article['Content'].str.len()
user_article['Ideal_Time'] = user_article['Total_Words']/(wpm)
user_article['Portion_read']= np.where(user_article['Ideal_Time']>= user_article['Time Spent'],
                                       (user_article['Time Spent'])/user_article['Ideal_Time'],
                                       (user_article['Ideal_Time']/(user_article['Time Spent'])))
user_article.head()


# In[119]:

#Keep only relevant column which will be required in the modeling later on.
#Here portion read values present the percentage of the article read by the user, like 1st user read 
#only 1.6% of the full article and user 2 read 75% of the article.
user_doc = user_article[['User ID','Article_Id','Portion_read']]
user_doc.head()


# In[120]:

#Transpose the data to create user-doc matrix form.
user_doc = user_doc.pivot(index='User ID',columns = 'Article_Id',values = 'Portion_read')
user_doc = user_doc.fillna(0)
user_doc.head()


# In[121]:

# Create extra columns in the user-doc equal to the total number of article available.
for i in range(len(article_data['Content'])):
    if i not in user_doc.columns:
        user_doc[i]=0
#    else:
#        user_doc[i]=0
user_doc.head()
# we need to re-order the columns


# In[122]:

cols = list((user_doc.columns.values))
cols.sort()
cols[0:10]


# In[123]:

user_doc = user_doc.reindex(columns = cols)
user_doc.head()


# In[124]:

user_doc[4:5][4718]


# # User-Document – k*m
# # k Users and m Documents.
# 

# In[125]:

user_doc_df=user_doc.fillna(0)
user_doc_df.head()


# In[126]:

user_doc_df[4:5][4718]


# In[127]:

user_document_array = user_doc.as_matrix()
user_document_array


# # Document Matrix - m * n 
# # m Documents and n Features
# 

# In[128]:

dtm_df.head()


# In[129]:

dtm_df.iloc[dtm_df['zoya'].nonzero()]


# In[130]:

dtm_array


# # Final Matrix: User-Feature Matrix (k*n)
# # User-Features Matrix(k*n) = User-Document Matrix(k*m) . Document-Feature Matrix(m*n)

# In[131]:

user_feature_df = user_doc_df.dot(dtm_df)
user_feature_dfl = user_feature_df
user_feature_df.head()
#validated using exact values and its correct


# # Array representation of User_feature

# In[132]:

user_feature_array = np.matmul(user_document_array,dtm_array)
user_feature_array


# # Cosine similarity between users from user_feature

# In[133]:


s = pairwise_distances(user_feature_df,metric ='cosine')
pd.DataFrame(s)


# # Model for nearest neighbors

# In[134]:


neighbors = NearestNeighbors(n_neighbors=10,metric ='cosine')
neighborsfit = neighbors.fit(user_doc_df)
neighborsfit


# # finding 10 nearest neighbors of users from the user-feature matrix

# In[135]:

nusers = []
for i in range(user_doc_df.shape[0]):
    nnusers = neighbors.kneighbors(user_doc_df[i:i+1],11,return_distance=False)
    nnuserslist = nnusers.tolist()
    nusers.append(nnuserslist)
nusers_df = pd.DataFrame(nusers)
nusers_df


# # naming the neighbors of each user and dropping off the self user.

# In[136]:

nusers_df= pd.DataFrame(nusers_df[0].values.tolist(),columns=['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10'])
nusers_df.drop(['n0'],axis=1, inplace=True)
#nusers_df


# In[137]:

user_doc_df.head()


# # create Lambda for each user in the database
# # lambda = 1/(1+exp(-no of documents read by any target user))

# In[138]:

#Lambda_user is the number of document read by any user, and this number would be used for target user.

no_document_read = user_doc_df.astype(bool).sum(axis=1)
no_document_read
lambda_all_user =[]
for i in no_document_read:
    l = 1/(1+math.exp(-i))
    lambda_all_user.append(l)
lambda_all_user[0:10]


# # join the lamda value with the user_feature_dfl

# In[139]:

user_feature_dfl['lambda']= lambda_all_user
user_feature_dfl.head()


# In[140]:

user_feature_df.drop(['lambda'],axis=1,inplace=True)
user_feature_df.head()


# # Create Average feature value for all users

# In[141]:


feature_sum_all = user_feature_df.sum(axis=0)
feature_count = user_feature_df.astype(bool).sum(axis=0)
avg_all_feature = feature_sum_all/feature_count
avg_all_feature=avg_all_feature.fillna(0)

#avg_all_feature


# In[142]:


avg_all_list= avg_all_feature.tolist()
avg_all_list[-1]
len(avg_all_list)


# In[143]:

blf = pd.DataFrame(avg_all_feature,columns=['Avg_all_values'])
blf.index.name = ' '
blf = blf.transpose()
#blf.drop(['lambda'],axis=1,inplace=True)
blf.head()


# In[144]:

feature_names=list(user_feature_df.columns.values)
len(feature_names)


# In[145]:

user_feature_dfl['lambda']= lambda_all_user
user_feature_dfl.head()


# In[146]:

user_feature_dfl.head()


# In[147]:

dd=[]
for index, row in user_feature_dfl.iterrows():
    list1 =[k*(1-row['lambda']) for k in avg_all_feature ]
    dd.append(list1)
dd


# In[148]:

avgall_user_feature = pd.DataFrame(dd)
#avgfeature_allusers.drop([avgfeature_allusers.shape[1]-1],axis=1,inplace=True)
avgall_user_feature.head()


# In[149]:

avgall_user_feature.shape[1]


# # we have average feature of all user - avgfeature_alluser
# # avgfeature_alluser = (1-lambda)avgfeature

# In[150]:

avgall_user_feature.columns=feature_names
#avgall_user_feature.head()


# # Now create Target user feature
# # target_user_feature = lambda * uaer_feature_df

# In[151]:

user_feature_df.head()


# In[152]:

target_user_feature = user_feature_df.mul(user_feature_df['lambda'],axis=0)
target_user_feature.drop(['lambda'],axis=1,inplace=True)
#target_user_feature.head()


# # now create a final_target_feature
# # final_target_feature = avgall_user_feature + target_user_feature

# In[153]:

avgall_user_feature.index +=1
avgall_user_feature.head()


# In[154]:

target_user_feature.head()


# In[155]:

final_target_feature = avgall_user_feature.add(target_user_feature,fill_value=0)
final_target_feature.head()
#verified and tested


# In[156]:

target_user_feature.shape


# In[157]:

dtm_df.shape


# In[158]:


neighbors = NearestNeighbors(n_neighbors=10,metric ='cosine')
neighborsfit = neighbors.fit(dtm_df)
neighborsfit


# In[159]:

nusers = []
for i in range(final_target_feature.shape[0]):
    nnusers = neighbors.kneighbors(final_target_feature[i:i+1],11,return_distance=False)
    nnuserslist = nnusers.tolist()
    nusers.append(nnuserslist)
nusers_df = pd.DataFrame(nusers)
#nusers_df


# In[160]:

nusers_df1= pd.DataFrame(nusers_df[0].values.tolist(),columns=['Article-1st','Article-2nd','Article-3rd','Article-4th','Article-5th','Article-6th','Article-7th','Article-8th','Article-9th','Article-10th','Article-11th'])
#nusers_df.drop(['n0'],axis=1, inplace=True)
#nusers_df1


# In[161]:

def nneighbors(target):
    neighbors = NearestNeighbors(n_neighbors=10,metric ='cosine')
    neighborsfit = neighbors.fit(dtm_df)
    neighborsfit
    nusers = []
    for i in range(target.shape[0]):
        nnusers = neighbors.kneighbors(target[i:i+1],11,return_distance=False)
        nnuserslist = nnusers.tolist()
        nusers.append(nnuserslist)
    nusers_df = pd.DataFrame(nusers)
    nusers_df1= pd.DataFrame(nusers_df[0].values.tolist(),columns=['Article-1st','Article-2nd','Article-3rd','Article-4th','Article-5th','Article-6th','Article-7th','Article-8th','Article-9th','Article-10th','Article-11th'])
    #nusers_df.drop(['n0'],axis=1, inplace=True)
    nusers_df1.index +=1
    return(nusers_df1)

target_feature = nneighbors(target = final_target_feature)
target_feature


# In[162]:

def old_specific_user_recommendation(userid):
    return target_feature.loc[userid]

def for_all_existing_user_recommendation():
    return target_feature


# In[163]:

#for the simiplicity sake we assume visitor Id is the user_id for new customer and if it exist in the database then it is old user and if not
#then it gets added and at the same time few articles gets presented and data gets stored.
userid = input("Please Enter the user id for which recommendation is required.")
if int(userid) in clickstream['User ID'].unique().tolist():
    print("User already exist, so the recommendations for user "+userid+ "are below:")
    specificuser = old_specific_user_recommendation(int(userid))
    print(specificuser)
else:
    print("Please wait we are fetching the recommendation for new user "+userid+" and they are ")
    new_recommendation = new_user_recommendation()
    print(new_recommendation.tolist())

foralluser = input("Do you want to see recommendations for all existing users then type YES/NO : ")
if foralluser.lower()=="yes":
    alluser = for_all_existing_user_recommendation()
    print(alluser)

print("Thank You for using our recommendation system, we are waiting for you.")

