import pandas as pd
import matplotlib.pyplot as plt

# Read the csv file 
df = pd.read_csv('iris_csv.csv')

#plot sepallength on x axis vs sepalwidth on y axis
x = df['sepallength']
y = df['sepalwidth']
print(df)

#plot sepal length on x asis as red and sepal width on y axis as blue
plt.scatter(x,y)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Sepal Length vs Sepal Width')
plt.show()

pd.plotting.scatter_matrix(df)
plt.show()