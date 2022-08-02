#Pandas has so many uses that it might make sense to list the things it can't do instead of what it can do.
#This tool is essentially your data’s home. Through pandas, you get acquainted with your data by cleaning, transforming, and analyzing it.
#For example, say you want to explore a dataset stored in a CSV on your computer. Pandas will extract the data from that CSV into a DataFrame — a table, basically — then let you do things like:

#Calculate statistics and answer questions about the data, like
#What's the average, median, max, or min of each column?
#Does column A correlate with column B
#What does the distribution of data in column C look like
#Clean the data by doing things like removing missing values and filtering rows or columns by some criteria
#Visualize the data with help from Matplotlib. Plot bars, lines, histograms, bubbles, and more.
#Store the cleaned, transformed data back into a CSV, other file or database
#Before you jump into the modeling or the complex visualizations you need to have a good understanding of the nature of your dataset and pandas is the best avenue through which to do that.

import numpy as np
import pandas as pd

#We can create a DataFrame from Python lists and Dictionaries but typically data is loaded into a DataFrame from Databases or CSV files and the coolest 
# part is it can also read URLS, which is nicer than having to download large files to your system.
#Let’s take a simple dataset that describe 3 species of Iris flowers and use it to learn pandas:

df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

#the function .head() displays the first few rows of the data so we can get a taste of the structure.

#Just like in NumPy we can use shape to tell us how large our data is.

#df.shape


#The info() method of pandas.DataFrame 
# can display information such as the number of rows and columns, 
# the total memory usage, the data type of each column, and the number of non-NaN elements.

df.info()