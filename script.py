import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import numpy as np

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
        plt.scatter(tennis['win_percentage'], tennis[column])
        plt.xlabel('Win Percentage')
        plt.ylabel('{}'.format(column))
        plt.title('{} vs. Win Percentage'.format(column))
        plt.show()
        plt.clf()
    else:
        pass'''
from scipy.stats import pearsonr
from scipy import stats
pearson_score, x= stats.pearsonr(tennis['Ranking'], tennis.win_percentage)
print(tennis.info())
print(pearson_score)
pearson, f = stats.pearsonr(tennis['total_matches'], tennis['win_percentage'])
print(pearson)

pearsonr, x = stats.pearsonr(tennis['win_percentage'], tennis['FirstServePointsWon'])
print(pearsonr)










## perform single feature linear regressions here:
fpw = np.array(tennis['Wins`']).reshape(-1, 1)
ranking = np.array(tennis['Ranking']).reshape(-1, 1)
fpw_train, fpw_test, ranking_train, ranking_test = train_test_split(fpw, ranking, test_size=0.2)
regr = LinearRegression()
regr.fit(fpw_train, ranking_train)
score_fpw_ranking = regr.score(fpw_test, ranking_test)
print('\n', score_fpw_ranking)





















## perform two feature linear regressions here:






















## perform multiple feature linear regressions here:
