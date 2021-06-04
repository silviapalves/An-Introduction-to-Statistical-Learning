import pandas as pd
import matplotlib.pyplot as plt

college = pd.read_csv(r"C:\Users\spalves\Desktop\Silvia\Pessoal\ITAU\ALL+CSV+FILES\ALL CSV FILES\College.csv")

college.dropna() 
print(college.describe())
print(college.count())

college.boxplot(column=['Outstate'])
college.hist(column=['Outstate'],bins=3)
college.hist(column=['Outstate'],bins=10)
college.hist(column=['Outstate'],bins=20)
