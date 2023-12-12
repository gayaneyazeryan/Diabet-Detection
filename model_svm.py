import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


df=pd.read_csv('diabetes_prediction_dataset.csv')
df.drop_duplicates(inplace=True)
df1 = df.drop('diabetes', axis = 1)


numeric_col=[]
non_numeric_col=[]
for column in df1.columns:
    if pd.api.types.is_numeric_dtype(df1[column]):
        if(df1[column].nunique()<5):
            non_numeric_col.append(column)
        else:
            numeric_col.append(column)
    else:
        non_numeric_col.append(column)

def univariate_analysis_cat(col):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Countplot
    sns.countplot(x=df[col],data=df , ax=ax[0])
    ax[0].set_title(f'Countplot for {col}')

    # Pie plot
    data_counts = df[col].value_counts()
    ax[1].pie(data_counts, labels=data_counts.index, autopct='%1.3f%%', startangle=90, colors=sns.color_palette('pastel'))
    ax[1].set_title(f'Pie plot for {col}')

    # plt.show()

for col in non_numeric_col:
    print(f' Univariate analysis for {col} column:')
    univariate_analysis_cat(col)

sample_size = 30000
sample_df = df.sample(n=sample_size, random_state=42)
age_group = pd.cut(df['age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80])


# print(sample_df['age'].mean())
# print(df['age'].mean())


le=LabelEncoder()
for col in non_numeric_col:
    sample_df[col]=le.fit_transform(sample_df[col])

x = sample_df.drop('diabetes', axis=1)
y = sample_df['diabetes']

#train model

from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
model_svm = SVC(kernel = 'linear', random_state = 0)
model_svm.fit(X_train, y_train)

y_pred = model_svm.predict(X_test)

accuracy_svm = accuracy_score(y_test, y_pred)
print(accuracy_svm)


classification_rep_svm = classification_report(y_test, y_pred)
