# MexEE 402: Data Preprocessing Case Study

# MexEE Elective 2: Data Science and Machine Learning
# Batangas State University, Alangilan Campus
# 1st Semester, AY 2026-2027

## Members

| Name | Student Number | Section |
|---|---|---|
| Aguila, Carl Michael |23-05482| MEXE- 4102 |
| Solis, John King Louies |23-02601| MEXE 4102 |

## Notebook links

| Chapter |Aguila, Carl Michael & Solis John King Louies | 
|---|---|
| Ch1_2_3 | [link](https://colab.research.google.com/drive/1NRGjfqWyvwN59f9tkFbYMcpEFJu6EcHr?usp=sharing) |
| Ch4 | [link](https://colab.research.google.com/drive/1IswrPRlqrb6R4PHdbww1L5Hcy0nMJnRV?usp=sharing) | 
| Ch5 | [link](https://colab.research.google.com/drive/1K7LKp3KNuxQ5TPIDdS83HA8-XPq2gxaA?usp=sharing) | 
| Ch6 | [link](https://colab.research.google.com/drive/1XFHDCh2x3favgJYNl81uoDcc1kr6hE0D?usp=sharing) | 
| Ch7 | [link](https://colab.research.google.com/drive/1AXL6tuo9lMvIj4i5rrF-KUocznB1hDv6?usp=sharing) | 
| Ch8 | [link](https://colab.research.google.com/drive/14ypR8IXPuT_4UUOtXS3w-zgWjcYgo9UZ?usp=sharing) | 
| Ch9 | [link](https://colab.research.google.com/drive/1JXfN5HHaz9lI8hTPUjt-yjca5kjAv5ik?usp=sharing) |

## What we learned

  In Chapters 1–3, we learned how to explore and understand a dataset using pandas, where head(), info(), and describe() help us check the data, understand its structure, and view its statistics. What surprised us was that missing values and unnecessary columns can affect how we understand and analyze the data. In Chapter 4, we learned that feature engineering makes data more useful by creating new features, grouping numerical values into categories, and encoding categorical data. What surprised us was that simple changes, like calculating lemonade sold per degree of temperature, can reveal relationships that are not obvious from the original data. In Chapter 5, we learned that preprocessing includes scaling values to make features comparable and identifying outliers that may affect analysis, and we were surprised by how numerical ranges and unusual values can influence results. In Chapter 6, we learned that outliers are values that differ greatly from most of the data and that different methods can detect and handle them, but removing them is not always the best choice because it can cause information loss. In Chapter 7, we learned that feature selection identifies the most useful features for predictions, and we were surprised that keeping every feature is not always better because irrelevant features can reduce a model’s accuracy. In Chapter 8, we learned how preprocessing pipelines combine steps like filling missing values and scaling numerical data into one organized process, and we were surprised that automation can make data preparation more consistent and reduce mistakes. Finally, in Chapter 9, we learned how to apply preprocessing techniques to a real-world Titanic dataset, including handling missing values, scaling numbers, encoding categories, and grouping ages. What surprised us was how many steps are needed to prepare raw data properly before it can be used for analysis or machine learning.

## Errors we found
# Chapter 1, 2, & 3 — Data preprocessing and cleaning
**1. Using inplace=True when filling missing values**

**Original code:**
```
df['Year'].fillna(df['Year'].mean(), inplace=True)
df['Publisher'].fillna(df['Publisher'].mode()[0], inplace=True)
```
**Problem:** The notebook already produces a FutureWarning because this way of modifying a column may not work as expected in pandas 3.0.

**Recommended fix:**
```
df['Year'] = df['Year'].fillna(df['Year'].mean())
df['Publisher'] = df['Publisher'].fillna(df['Publisher'].mode()[0])
```
**2. Removing missing publishers after filling them**

**Original code:**
```
df = df[df['Publisher'].notna()]
```
**Problem:** This line appears after the missing Publisher values have already been filled with the mode. Therefore, it will normally remove no rows and is redundant.

**Recommended fix:** If the goal is to fill missing publishers, remove the unnecessary filtering line. If the goal is to delete rows with missing publishers instead, use dropna() before imputing those values.

## Chapter 6 — Outlier detection
**1. Z-score does not identify the unusual value**

**Original code:**
```
z_scores = stats.zscore(data)
outliers = data[np.abs(z_scores) > 3]
```
**Problem:** The sample value 100 has a Z-score of approximately 2.62, so the threshold of 3 does not identify it.

**Alternative approach:**
```
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

outliers = data[
    (data < Q1 - 1.5 * IQR) |
    (data > Q3 + 1.5 * IQR)
]
```
The IQR method identifies 100 as an outlier in this example.

**Important:** The original Z-score code is valid. This is a limitation of the chosen method and threshold for this sample, not a definite coding error.

## Chapter 7 — Feature selection
**1. The target variable is included in the correlation-based feature list**

**Original code:**
```
correlations = df_2.corr()['final grade'].sort_values()
relevant_features = correlations[correlations > 0.5]
```
**Problem:** The results include final grade itself with a correlation of 1.0. If the list is intended to contain input features for predicting final grades, the target must not be included as an input feature.

**Recommended fix:**
```
correlations = df_2.corr()['final grade'].drop('final grade')
relevant_features = correlations[correlations > 0.5]
```
**2. Too few samples for five-fold cross-validation**

**Original code:**
```
selector = RFECV(estimator, step=1, cv=5)
```
**Problem:** The example has only seven rows. With five-fold cross-validation, some validation folds contain only one sample, causing an UndefinedMetricWarning because the R^2
 score cannot be calculated reliably with fewer than two samples.

**Recommended fix:** Use a larger dataset with enough samples for cross-validation. If you must use the small demonstration dataset, choose a validation strategy appropriate to the sample size, while recognizing that the results will remain unreliable.

## Chapter 8 — Constructing a preprocessing pipeline
**1. The preprocessing pipeline only keeps Age and Fare**

**Original code:**
```
preprocessor = ColumnTransformer(transformers=[
    ('age_fare', pipeline, ['Age', 'Fare'])
])
X_transformed = preprocessor.fit_transform(X)
```
**Problem:** The original X contains other columns, but the ColumnTransformer processes only Age and Fare. By default, the other columns are dropped from the transformed output.

This is not necessarily an error if the purpose is only to demonstrate scaling and imputation on two columns. However, it is incomplete if the goal is to preprocess the full Titanic dataset.

**Recommended fix:** Add preprocessing for the categorical columns, such as Sex, Embarked, and Pclass, using an imputer and OneHotEncoder. Alternatively, explicitly set remainder='passthrough' if the remaining columns are already suitable for the model.

## Chapter 9 — Real-world Titanic preprocessing
**1. The age column is discretized after preprocessing**

**Original code:**
```
titanic_preprocessed = preprocessor.fit_transform(data)

bins = [0, 12, 50, 200]
labels = ['Child', 'Adult', 'Elderly']
data['Age'] = pd.cut(data['Age'], bins=bins, labels=labels)
```
**Problem:** The preprocessing pipeline is fitted and transformed before the age categories are created. Therefore, titanic_preprocessed still contains the earlier transformed age values, not the new Child, Adult, and Elderly categories.

**Recommended fix:** Create the age categories before fitting the preprocessor if you intend to use those categories as input features. Then configure the preprocessing pipeline to encode the categorical age feature appropriately.

**2. The age-distribution plot uses the wrong data**

**Original code:**
```
plt.hist(data['Age'].dropna(), alpha=0.5,
         label='Before discretization')

plt.hist(titanic_preprocessed[:, 2], alpha=0.5,
         label='After discretization')
```
**Problem:** By the time this plotting cell runs, data['Age'] has already been changed to categories, so it is not the original age distribution. Also, titanic_preprocessed[:, 2] refers to the third transformed feature, not necessarily age.

**Recommended fix:** Save the original age values before discretization, and plot the actual age-category counts after discretization.
```
age_before = original_data['Age'].dropna()

age_after = pd.cut(
    original_data['Age'],
    bins=[0, 12, 50, 200],
    labels=['Child', 'Adult', 'Elderly']
)

age_after.value_counts().reindex(
    ['Child', 'Adult', 'Elderly']
).plot(kind='bar')
```
**3. The preprocessing pipeline is fitted before a train/test split**

**Problem:** The notebook fits the preprocessing transformers on the full dataset. If this is used for predictive modeling, statistics learned from the test data can influence preprocessing.

**Recommended fix:** Split the dataset into training and test sets first, fit the preprocessing pipeline on the training features only, and use transform() on the test features.

## Note on AI tools

We used ChatGPT to better understand data science concepts, organize the structure of our Markdown summaries, and improve the clarity of our written responses.

With the help of these tools, we also gained a better understanding of the mathematical concepts involved in machine learning pipelines and cross-validation, especially when working with small datasets.

## References

* McKinney, W. (2021). Python for Data Analysis, 3rd ed. O'Reilly.
* VanderPlas, J. Python Data Science Handbook.

