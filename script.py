import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

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






















## perform single feature linear regressions here:






















## perform two feature linear regressions here:






















## perform multiple feature linear regressions here:
