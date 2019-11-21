import pandas as  pd
import numpy as np
import re
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.decomposition import PCA
from datetime import datetime
from datetime import date
import pymysql
pymysql.__version__
'0.9.3'

# Mysql Server connection
mysql_conn = pymysql.connect(host='35.230.132.104',
database='low_carb_program_v2',
user='krishP',
password='s4lv4d0r',)

# Data
users = pd.read_sql_query('''SELECT* FROM td_user''', mysql_conn)

data = pd.read_csv('diab_risk.csv', encoding= 'utf-8')
data = pd.merge(data,users, left_on='id', right_on='email')

data = data[['What gender are you?', 'What is your ethnic group?',
       'What is your occupational status?', 
       'What is your household income?',
       'Which type of diabetes do you have?',
       'How many grams of carbs do you eat daily (approximately to the nearest gram)?',
       'Intense hunger', 'Tiredness', 'Difficulty concentrating',
       'Spikes in blood glucose levels', 'Mood swings', 'Irritability',
       'Anxiety',
       'What is your height?', 'What is your waist circumference?',
       'What is your current weight in stones and pounds?', 'Unnamed: 24',
       'What is your weight in kilograms (to the nearest kilogram)?',
       'dob']]




data.columns = ['gender', 'ethnic group',
       'occupational status', 'income','diabetes_type',
       'daily carbs',
       'intense hunger', 'tiredness', 'difficulty concentrating',
       'spikes in blood glucose levels', 'mood swings', 'irritability',
       'anxiety',
       'height', 'waist',
       'stones', 'pounds',
       'weight_kg',
       'dob']


data.isnull().sum() #null values in each column of the DataFrame 


# weights in Kgs from stones and pounds
def stones_pounds(data):
    # stones
    data = data.reset_index()
    stones = data[['stones']]
    stones = stones.reset_index()

    # preprocess weight
    pounds = data[["pounds"]]
    pounds.columns =['pounds']

    df = pd.concat([stones,pounds],axis=1)
    df = df.drop(df.index[0])
    df = df.reset_index()
    df = df.dropna() 
    df = df.astype(int)
    # concat stones and pounds
    df['new_weights'] = df['stones'].astype(str)+"."+df['pounds'].astype(str)
    df =df.drop(columns=['stones','pounds'])
    df = df.astype(float)
    df['new_weights'] = (df['new_weights']*6.4).round()
    df = df.drop(columns=['level_0','index'])
    return df
df = stones_pounds(data)

# filling nans with converted weights 
data['weight_kg'] = data.weight_kg.fillna(df.new_weights)
data = data.drop(columns=['stones','pounds'])
data = data.dropna()

def preprocess_height(data):

    data['height'] = data['height'].str.split('/').str[1]
    data['height'] = data['height'].str.replace(r'm','')
    return data
data = preprocess_height(data)

def preprocess_waist(data):

    data['waist'] = data['waist'].str.split('/').str[1]
    data['waist'] = data['waist'].str.replace(r'cm','')
    return data

data = preprocess_waist(data)
data = data.dropna()


def cal_age(data):


    data['age'] = data['dob'].apply(lambda x: x.year)
    data['age'] = 2019 - data['age']

    return data
data = cal_age(data) # we can remove outliers from the age

final_df = data
def encoder(final_df):

    final_features = final_df.drop(columns=[
        # 'index',
        'daily carbs','height','waist','weight_kg','age'])
    encoder = LabelEncoder()
    # encoding features
    final_features = final_features.apply(LabelEncoder().fit_transform)
    df = final_df[['daily carbs','height','waist','weight_kg','age']]
    final_df = pd.concat([final_features,df],axis=1)

    return final_df
encoded_df = encoder(final_df)
# encoded_df['daily carbs'] = scaler.fit_transform(encoded_df['daily carbs'])

def scale(encoded_df):
    # feature mattrix
    scaler = MinMaxScaler()
    x = encoded_df.drop(columns=['diabetes_type'])
    x = pd.DataFrame(scaler.fit_transform(x))

    return x
x = scale(encoded_df)

# correlation mattrix
corr = x.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(4)


# df with column names
corr_df = encoded_df.corr()
corr_df.style.background_gradient(cmap='coolwarm').set_precision(4)

# label
y = encoded_df[['diabetes_type']]

# idex2label
Labels = final_df['diabetes_type'].values
ids = encoded_df['diabetes_type'].values
idx2intent = {i:j for i, j in zip(ids, Labels)}
idx2intent[0]#2, 5, 6, 7
# ignoring healthcare, mody, getational diabetes

x = x.reset_index()
y = y.reset_index()
features = pd.concat([x,y],axis=1)
no_diab = features[features['diabetes_type']==3]
no_diab_dist = no_diab[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]
no_diab_y = no_diab[['diabetes_type']]
prediab = features[features['diabetes_type']==6]
type1 = features[features['diabetes_type']==7]
type2 = features[features['diabetes_type']==8]

final_features = pd.concat([no_diab, prediab, type1, type2])
x = final_features[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]
y = final_features[['diabetes_type']]


# gaussian distribution plot
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
mu = 0
variance = 1
sigma = math.sqrt(variance)
plt.plot(no_diab_dist[3], norm.pdf(no_diab_dist[3],mu,sigma))
plt.show()


# Bivariate Gaussian

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D



# Our 2-dimensional distribution will be over variables X and Y
N = 60
# X = np.linspace(-3, 3, N)
X = no_diab_dist
Y = no_diab_y
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix
mu = np.array([0., 1.])
Sigma = np.array([[ 1. , -0.5], [-0.5,  1.5]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

# The distribution on the variables X, Y packed into pos.
Z = multivariate_gaussian(pos, mu, Sigma)

# Create a surface plot and projected filled contour plot under it.
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                cmap=cm.viridis)

cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)

# Adjust the limits, ticks and view angle
ax.set_zlim(-0.15,0.2)
ax.set_zticks(np.linspace(0,0.2,5))
ax.view_init(27, -21)

plt.show()



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.3)


clf = SVC(gamma='auto',verbose=True)
model = clf.fit(x_train, y_train) 
y_pred = model.predict(x_test)
accuracy_score(y_test,y_pred)*100
confusion_matrix(y_test,y_pred)
# recall_score(y_test, y_pred,pos_label='positive',average='micro')
f1_score(y_test, y_pred, average='macro')
mean_squared_error(y_test, y_pred)

y_true = y_test
y_pred = y_pred
cnf_matrix = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plt.show()
plot_confusion_matrix(cnf_matrix, classes=['no_diab', 'prediab', 'type1','type2'],
                      title='Confusion matrix, without normalization')



