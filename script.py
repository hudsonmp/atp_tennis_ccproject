import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_size import AxesX
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
scaler = StandardScaler()
mmscaler = MinMaxScaler()
# load and investigate the data here:

tennis = pd.read_csv('/Users/hudsonmitchell-pullman/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Learning to Code/Codecademy/ML:AI Engineer/tennis_ace_starting/tennis_stats.csv')
max_columns = 1000
max_rows = 100
pd.set_option('display.max_columns', max_columns)
pd.set_option('display.max_rows', max_rows)
print(tennis.head())
print(tennis.info())
print(tennis.describe(include='all'))



# perform exploratory analysis here:
plt.scatter(tennis['Ranking'], tennis['DoubleFaults'])
plt.xlabel('Ranking')
plt.ylabel('DoubleFaults')
plt.title('Double Faults vs. Ranking')
#plt.show()
#plt.clf()
tennis['total_matches'] = tennis['Wins'] + tennis['Losses']
tennis['win_percentage']  = np.where(tennis['total_matches'] > 0, tennis['Wins'] / tennis['total_matches'], 0)
'''for column in tennis.columns:
    #print(tennis[column].dtype)
    if tennis[column].dtype != 'object':
        plt.scatter(tennis['Ranking'], tennis[column])
        plt.xlabel('Ranking')
        plt.ylabel('{}'.format(column))
        plt.title('{} vs. Ranking'.format(column))
        plt.show()
        plt.clf()
    else:
        pass'''
from scipy.stats import pearsonr
from scipy import stats
'''pearson_score, x= stats.pearsonr(tennis['Ranking'], tennis.win_percentage)
print(tennis.info())
print(pearson_score)
pearson, f = stats.pearsonr(tennis['total_matches'], tennis['win_percentage'])
print(pearson)

pearsonr, x = stats.pearsonr(tennis['win_percentage'], tennis['FirstServePointsWon'])
print(pearsonr)'''

'''pearsonr, p = 0,0
for column in tennis.columns:
    if tennis[column].dtype != 'object':
        pearsonr, x = stats.pearsonr(tennis[column], tennis['Ranking'])
        if -0.3 >= pearsonr or pearsonr >= 0.3:
            print('The Pearson score for {} is {}'.format(column, pearsonr))'''
## perform single feature linear regressions here:

print('\n', '\n')
tennis_new = tennis.select_dtypes(include=['int64', 'float64'])
correlation_matrix = tennis_new.corr()
plt.figure(figsize=(15, 13))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidth=0.5, annot_kws={"size": 8})
plt.title("Correlation Matrix")
plt.show()
x = np.array(tennis_new['TotalServicePointsWon'])
y = np.array(tennis_new['Wins'])
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y)
lgr = LinearRegression().fit(x_train, y_train)
tennis1_y_pred = lgr.predict(x_test)
tennis1_score = lgr.score(x_test, y_test)
#print(tennis1_y_pred)
print(tennis1_score)
## perform two feature linear regressions here:


## perform multiple feature linear regressions here:
y = tennis_new[['TotalServicePointsWon', 'ServiceGamesWon', 'SecondServePointsWon', 'FirstServePointsWon']]
X = tennis_new[['TotalPointsWon']]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
lgr = LinearRegression()
lgr.fit(X_train, y_train)
score_mult = lgr.score(X_test, y_test)
print(score_mult)