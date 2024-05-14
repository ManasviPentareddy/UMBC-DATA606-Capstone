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
### Histogram for 'CRIM', 'NOX', 'RM', 'AGE', 'MEDV'.

<img width="835" alt="image" src="https://github.com/ManasviPentareddy/UMBC-DATA606-Capstone/assets/157778795/cd40d4ad-39eb-4d6c-94b5-d8b61985b830">

The histograms provide insights into the distributions of the selected features within the dataset.

From the KDE lines, you can observe the general shape and smoothness of each distribution, indicating the density of data points across different ranges of each feature.

These visualizations help in understanding the central tendency, spread, and skewness of each feature's distribution, which can be crucial for further analysis and decision-making in this project.

### Calculate and print mean,median,mode, minimum and maximum values for each feature
<img width="230" alt="image" src="https://github.com/ManasviPentareddy/UMBC-DATA606-Capstone/assets/157778795/882ed1a6-a6c9-4b52-bc74-9ab9db20f3b6">

<img width="230" alt="image" src="https://github.com/ManasviPentareddy/UMBC-DATA606-Capstone/assets/157778795/c209f83c-9de5-412b-8e79-30b248c38133">

It gives an idea of the typical value for each feature, useful for understanding their overall magnitude,gives insight into the central position,helps identify the most common values,lower and upper bound and revealing the range of values and showing the maximum extent of variation respectively.

### Showing 'CHAS' value counts by using value counts and bar plot

<img width="207" alt="image" src="https://github.com/ManasviPentareddy/UMBC-DATA606-Capstone/assets/157778795/76b17edd-7a74-4a7a-9ef8-bb4943beb8c9">

<img width="344" alt="image" src="https://github.com/ManasviPentareddy/UMBC-DATA606-Capstone/assets/157778795/144bac34-97d5-4424-bfa5-f8e3e43474cd">

The value counts provide a numeric summary of the distribution of values in the 'CHAS' column, while the seaborn bar plot offers a visual representation, making it easier to grasp the relative frequencies of different values.










