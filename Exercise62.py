import pandas as pd
import matplotlib.pyplot as plt
import sklearn.decomposition as SD
import sklearn.preprocessing as SP
import sklearn.cluster as SC

# Read the csv file

quantitive = ['sepallength','sepalwidth','petallength','petalwidth']

df = pd.read_csv('iris_csv.csv')
class_names = pd.Categorical(df['class']).rename_categories(['setosa','versicolor','virginica'])


df_quantitive = df[quantitive]

#standardize the data
df_standardized = SP.StandardScaler().fit(df_quantitive).transform(df_quantitive)

#create a PCA model 
pca = SD.PCA().fit(df_standardized)

#transform data into new space
df_pca = pd.DataFrame(pca.transform(df_standardized))

#add data into the extended dataframe
df_extended = pd.concat([df,df_pca],axis=1)

#df_extended.plot.scatter(0,1,c=class_names.codes,cmap='brg')

pca_loadings=pd.DataFrame(pca.components_, columns = df_quantitive.columns)
#print(pca_loadings)

#calculate the correlation between the original data features 
print(df_quantitive.corr('spearman'))

#calculate the correlation between the pca features
print(df_pca.corr('spearman'))

kmeans = SC.KMeans(n_clusters=3).fit(df_standardized)
predictions = kmeans.predict(df_standardized)
df_extended['cluster'] = pd.Categorical(predictions)


fig, ax = plt.subplots(2, 1, figsize=(10, 10))
df_extended.plot.scatter(0, 1, c=class_names.codes, cmap='brg', ax=ax[0])
ax[0].set_title('PCA')


df_extended.plot.scatter(0, 1, c='cluster', cmap='brg', ax=ax[1])
ax[1].set_title('Clusters')
plt.show()

print(pd.crosstab(df_extended['cluster'], class_names)) 
print("actual amount of setosa: ", sum(i for i in df['class'] == 'Iris-setosa'))
print("actual amount of versicolor: ", sum(i for i in df['class'] == 'Iris-versicolor'))
print("actual amount of virginica: ", sum(i for i in df['class'] == 'Iris-virginica'))

