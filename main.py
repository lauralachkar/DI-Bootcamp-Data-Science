#NumPy (Numerical Python) est une 
# bibliothèque de python qui comporte des 
# fonctions permettant de manipuler des matrices ou tableaux 
# multidimensionnels.NumPy est la base de SciPy, qui n’est rien d’autre qu’un ensemble 
# de bibliothèques Python pour des calculs scientifiques. Il est beaucoup plus adapté pour les 
# problématiques qui requièrent l’usage des matrices ou des tableaux multidimensionnels, comme l
# a Data Science, l’ingénierie, les mathématiques ou encore les simulations.Lors des calculs logiques et mathématiques sur des matrices et tableaux, 
# c’est NumPy qui est très sollicité. Il permet d’effectuer rapidement et efficacement les opérations par rapport aux listes Python.
#Les tableaux NumPy utilisent d’abord moins de mémoire et d’espace de stockage, ce qui le rend plus avantageux que les tableaux traditionnels de python.
#En effet, un tableau NumPy est de petite taille et ne dépasse pas les 4MB. Mais une liste peut atteindre les 20MB. De plus, les tableaux NumPy sont faciles à manipuler.

#numpy.arange([start, ]stop, [step, ]dtype=None, *, like=None)
#Return evenly spaced values within a given interval.
#arange can be called with a varying number of positional arguments:

#arange(stop): Values are generated within the half-open interval [0, stop) (in other words, the interval including start but excluding stop).
#arange(start, stop): Values are generated within the half-open interval [start, stop).
#arange(start, stop, step) Values are generated within the half-open interval [start, stop), with spacing between values given by step.

#The plot() function is used to draw points (markers) in a diagram.
#By default, the plot() function draws a line from point to point.
#The function takes parameters for specifying points in the diagram.
#Parameter 1 is an array containing the points on the x-axis.
#Parameter 2 is an array containing the points on the y-axis.
#If we need to plot a line from (1, 3) to (8, 10), we have to pass two arrays [1, 8] and [3, 10] 
# to the plot function.

#With Pyplot, you can use the title() function to set a title for the plot.

#With Pyplot, you can use the xlabel() and ylabel() functions to set a label for the x- and y-axis.


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Generation of variables 
x=np.arange(0,10) #Array of range 0 to 9
y=x**3

# Printing the variables
print(x)
print(y)


plt.plot(x,y) # Function to plot
plt.title('Line Chart') # Function to give title

# Functions to give x and y labels
plt.xlabel('X-Axis') 
plt.ylabel('Y-Axis')

# Functionn to show the graph  
plt.show()


#Multiple Line Chart

# Generation of 1 set of variables 
x = np.arange(0,11)
y = x**3

# Generation of 1 set of variables
x2 = np.arange(0,11)
y2 = (x**3)/2

# Printing all variables
print(x,y,x2,y2,sep="\n")

# "linewidth" is used to specify the width of the lines
# "color" is used to specify the colour of the lines
# "label"is used to specify the name of axes to represent in the lengend 
plt.plot(x,y,color='r',label='first data', linewidth=5) 
plt.plot(x2,y2,color='y',linewidth=5,label='second data')
plt.title('Multiline Chart')

# Uses the label attribute to display reference in legend
plt.ylabel('Y axis')
plt.xlabel('X axis')

# Shows the legend in the best postion with respect to the graph
#A legend is an area describing the elements of the graph. In the matplotlib library, 
# there’s a function called legend() which is used to Place a legend on the axes.
plt.legend()
plt.show()

#Bar Chart

# Generation of variables 
x = ["India",'USA',"Japan",'Australia','Italy']
y = [6,7,8,9,2]

# Printing the variables
print(x)
print(y)


#The matplotlib API in Python provides the bar() function which can be used in MATLAB style use or as an object-oriented API. The syntax of the bar() function to be used 
# with the axes is as follows:-
#plt.bar(x, height, width, bottom, align)

plt.bar(x,y, label='Bars1', color ='r') # Function to plot

# Function to give x and y labels 
plt.xlabel("Country")
plt.ylabel("Inflation Rate%")

# Function to give heading of the chart
plt.title("Bar Graph")

# Function to show the chart
plt.show()


#Multiple Bar Chart

x = ["India",'USA',"Japan",'Australia','Italy']
y = [6,7,8,9,5]

# Generation of 2 set of variables
x2 = ["India",'USA',"Japan",'Australia','Italy']
y2 = [5,1,3,4,2]

# Printing all variables
print(x,y,x2,y2,sep="\n")

# Functions to plot 
plt.bar(x,y, label='Inflation', color ='y')
plt.bar(x2,y2, label='Growth', color ='g')

# Functions to give x and y labels
plt.xlabel("Country")
plt.ylabel("Inflation & Growth Rate%")

plt.title("Multiple Bar Graph")
plt.legend()
plt.show()


#Histogram

# Generation of variable
stock_prices = [32,67,43,56,45,43,42,46,48,53,73,55,54,56,43,55,54,20,33,65,62,51,79,31,27]

# Function to show the chart
#Create a new figure, or activate an existing figure.
#The hist() function in pyplot module of matplotlib 
# library is used to plot a histogram.
plt.figure(figsize = (8,5))
plt.hist(stock_prices, bins = 5)


#Scatter Plot

# Generation of x and y variables
x = [1,2,3,4,5,6,7,8]
y = [5,2,4,2,1,4,5,2]

# Function to plot the graph
#With Pyplot, you can use the scatter() function to draw a scatter plot.
#The scatter() function plots one dot for each observation. 
# It needs two arrays of the same length, one for the values of the x-axis, and one for values on the y-axis:

plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot')


#Pie Chart

# Collection of raw data
raw_data={'names':['Nick','Sani','John','Rubi','Maya'],
'jan_score':[123,124,125,126,128],
'feb_score':[23,24,25,27,29],
'march_score':[3,5,7,6,9]}

# Segregating the raw data into usuable form/variables
#A Pandas DataFrame is a 2 dimensional data structure, like a 2 dimensional 
# array, or a table with rows and columns.

df=pd.DataFrame(raw_data,columns=['names','jan_score','feb_score','march_score'])
df['total_score']=df['jan_score']+df['feb_score']+df['march_score']

# Printing the data
print(df)

# Function to plot the graph
#With Pyplot, you can use the pie() function to draw pie charts:
#This function is used to set some axis properties to the graph.
plt.pie(df['total_score'],labels=df['names'],autopct='%.2f%%')
plt.axis('equal')
plt.axis('equal')
plt.show()


#Sub Plots


# Defining the sixe og the figures
plt.figure(figsize=(10,10))

# Generation of variables
x = np.array([1,2,3,4,5,6,7,8])
y = np.array([5,2,4,2,1,4,5,2])

# Generating 4 subplots in form of 2x2 matrix
# In the line below the arguments of plt.subplot are as follows:
# 2- no. of rows
# 2- no. of columns
# 1- position in matrix
# Position (0,0)

#With the subplot() function you can draw multiple plots in one figure:
plt.subplot(2,2,1)
plt.plot(x,y,'g')
plt.title('Sub Plot 1')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')

# Position (0,1)
plt.subplot(2,2,2)
plt.plot(y,x,'b')
plt.title('Sub Plot 2')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')

# Position (1,0)
plt.subplot(2,2,3)
plt.plot(y*2,x*2,'y')
plt.title('Sub Plot 3')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')

# Position (1,1)
plt.subplot(2,2,4)
plt.plot(x*2,y*2,'m')
plt.title('Sub Plot 4')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')

# Function for layout and spacing
plt.tight_layout(h_pad=5, w_pad=10)