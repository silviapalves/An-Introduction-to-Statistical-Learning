import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
import statsmodels.api as sm
from statsmodels.stats.api import anova_lm
from statsmodels.formula.api import ols
from scipy import stats
import numpy as np
from statsmodels.stats.outliers_influence import OLSInfluence

smarket = pd.read_csv(r"C:\Users\spalves\Desktop\Silvia\Pessoal\ITAU\ALL+CSV+FILES\ALL CSV FILES\Smarket.csv")
smarket.dropna() 
