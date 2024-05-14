# 1. House Price Prediction in Boston
Author - Manasvi Pentareddy
Semester - Spring'24
Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
GitHub profile : 
LinkedIn progile : 
PowerPoint :
YouTube video : 
# 2. Background


What is it about?
The Boston Housing dataset comprises various socio-economic attributes associated with housing in Boston. These attributes include factors such as crime rates, air quality, and access to highways, among others. The dataset is commonly used in predictive modeling tasks, particularly for estimating median house prices. By analyzing the relationships between these features and housing prices, machine learning algorithms can be trained to make accurate predictions, aiding in real estate valuation and urban planning efforts.

Why does it matter?
It helps users understand the significance of each piece of information, guiding them in focusing on the most relevant factors for their analysis or prediction tasks.

What are your research questions?
What factors influence housing prices in Boston suburbs?

Can we predict the median value of owner-occupied homes based on attributes such as crime rate, number of rooms, and proximity to highways?


# 3. Data
Describe the datasets you are using to answer your research questions.

Data sources : Kaggle https://www.kaggle.com/datasets/shubhammeshram579/house/data
Data size : 35.38
Data shape (number of rows and columns): 509 rows and 14 columns

<img width="844" alt="image" src="https://github.com/ManasviPentareddy/UMBC-DATA606-Capstone/assets/157778795/706fbcb6-dc31-40f0-939f-9e1f2d8bfa9c">

# 4. Exploratory Data Analysis
Found missing values in the dataset.
Filled them with mean method and displayed
Checked for duplicate values and found 4 values and deleted them.


# Visualizations
## Univariate Analysis
### i.Histogram for 'CRIM', 'NOX', 'RM', 'AGE', 'MEDV'.

<img width="835" alt="image" src="https://github.com/ManasviPentareddy/UMBC-DATA606-Capstone/assets/157778795/cd40d4ad-39eb-4d6c-94b5-d8b61985b830">

The histograms provide insights into the distributions of the selected features within the dataset.

From the KDE lines, you can observe the general shape and smoothness of each distribution, indicating the density of data points across different ranges of each feature.

These visualizations help in understanding the central tendency, spread, and skewness of each feature's distribution, which can be crucial for further analysis and decision-making in this project.

### ii.Calculate and print mean,median,mode, minimum and maximum values for each feature
<img width="230" alt="image" src="https://github.com/ManasviPentareddy/UMBC-DATA606-Capstone/assets/157778795/882ed1a6-a6c9-4b52-bc74-9ab9db20f3b6">

<img width="230" alt="image" src="https://github.com/ManasviPentareddy/UMBC-DATA606-Capstone/assets/157778795/c209f83c-9de5-412b-8e79-30b248c38133">

It gives an idea of the typical value for each feature, useful for understanding their overall magnitude,gives insight into the central position,helps identify the most common values,lower and upper bound and revealing the range of values and showing the maximum extent of variation respectively.

### iii.Showing 'CHAS' value counts by using value counts and bar plot

<img width="207" alt="image" src="https://github.com/ManasviPentareddy/UMBC-DATA606-Capstone/assets/157778795/76b17edd-7a74-4a7a-9ef8-bb4943beb8c9">

<img width="344" alt="image" src="https://github.com/ManasviPentareddy/UMBC-DATA606-Capstone/assets/157778795/144bac34-97d5-4424-bfa5-f8e3e43474cd">

The value counts provide a numeric summary of the distribution of values in the 'CHAS' column, while the seaborn bar plot offers a visual representation, making it easier to grasp the relative frequencies of different values.

# Bivariate Analysis
## i.Scatter plot of 'RM' and 'MEDV'.
<img width="430" alt="image" src="https://github.com/ManasviPentareddy/UMBC-DATA606-Capstone/assets/157778795/65c413f8-dce9-4563-b345-0f1cd19363a3">

A scatter plot of 'RM' (average number of rooms) vs. 'MEDV' (median house price) helps identify any linear relationship or pattern between these two continuous variables. For example, we can observe if there's a trend where houses with more rooms tend to have higher median prices.

Here by observing the scatter plot, you can assess the relationship between the average number of rooms and the median house price. Typically, in real estate, there might be a positive correlation between the number of rooms in a house and its price. As the number of rooms increases, the house price might also increase.

## ii.Pair plot for numerical variables like 'CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'MEDV'.
<img width="833" alt="image" src="https://github.com/ManasviPentareddy/UMBC-DATA606-Capstone/assets/157778795/0c8fb0b0-d7a6-495f-baa8-7ef395732846">

A pair plot visualizes pairwise relationships between multiple numerical variables in the dataset. Each scatter plot in the pair plot matrix shows the relationship between two variables. This helps identify potential correlations or trends between different features, allowing for a comprehensive exploration of the dataset.

### iii. Box plot for 'CHAS and 'MEDV'.
<img width="429" alt="image" src="https://github.com/ManasviPentareddy/UMBC-DATA606-Capstone/assets/157778795/4f117b8b-27d4-4820-aa1c-6f6775312134">

The box plot visualizes the distribution of median house prices ('MEDV') across two categories of Charles River proximity ('CHAS'). Each data point in the plot represents the median house price of a property, with properties near the Charles River (CHAS=1) depicted on one side of the plot, and those farther away (CHAS=0) on the other side.

## Multivariate Analysis
### i.3D Scatter plot of 'RM, 'AGE' and 'MEDV' with 'CHAS'.
<img width="404" alt="image" src="https://github.com/ManasviPentareddy/UMBC-DATA606-Capstone/assets/157778795/746a5de2-dace-4475-86ea-a693321267f8">

A 3D scatter plot visualizes relationships between three numerical variables simultaneously. In this plot, each point represents a property, with its position in 3D space determined by the values of 'RM' (average number of rooms), 'AGE' (age of the property), and 'MEDV' (median house price). Points are colored based on the value of the binary variable 'CHAS', providing insight into how proximity to the Charles River influences these relationships.

Relationship between RM, AGE, and MEDV:
Observing the distribution of points in 3D space, you might notice certain patterns or clusters. For example, there could be a cluster of points with higher 'RM' values (indicating more rooms), lower 'AGE' values (indicating newer properties), and higher 'Price' values (indicating higher house prices), suggesting a positive correlation between these variables.

### ii.Stacked bar plot of 'MEDV'' by 'CHAS'.
<img width="535" alt="image" src="https://github.com/ManasviPentareddy/UMBC-DATA606-Capstone/assets/157778795/c02791f5-e306-4ad8-bfb1-c120dcc26651">

The bars are color-coded to represent the two categories of proximity to the Charles River. Blue typically indicates properties far from the river ('CHAS = 0'), while salmon color represents properties near the river ('CHAS = 1').

Bar Heights: Each bar's height represents the mean median house price for properties in the respective category of river proximity. The taller the bar, the higher the average median house price for properties in that category.

## Outlier Detection
### i.Box plot of features with feature values
<img width="652" alt="image" src="https://github.com/ManasviPentareddy/UMBC-DATA606-Capstone/assets/157778795/207f7ec9-ee53-4f6f-8b57-3e0608e00ff8">

This horizontal box plot visualizes the distribution of feature values in the dataset, excluding the 'CHAS' column. It assists in identifying potential outliers and understanding the spread and central tendency of each feature.










