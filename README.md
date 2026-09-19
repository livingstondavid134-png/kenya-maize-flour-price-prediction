Kenya Maize Flour Price Prediction

Project Overview

This project uses multiple linear regression to predict the retail price of maize flour products in Kenya based on the product classification (brand/product) and date.

The project was developed as a practical machine learning project to apply data preprocessing, categorical encoding, feature engineering, regression modelling, and model evaluation to a real-world Kenyan dataset.

Objective

The main objective of this project is to investigate whether maize flour retail prices can be predicted using:

Product classification — the maize flour brand/product
Date — represented numerically as the number of days since the first observation

The target variable is the retail price of the maize flour product.

Dataset

The dataset contains retail price observations for different maize flour products in Kenya.

The dataset used for modelling contains:

* 1,415 observations
* 25 product classifications
* Date observations from 28 May 2025 to 28 December 2025
* Retail price as the target variable

Examples of product classifications in the dataset include:

* Dola
* Jogoo
* Ajab
* Hostess
* Soko
* Pembe
* Amaize
* Taifa
* Oryx
* Ndovu etc

The original dataset also contains information such as market, county, commodity, classification, retail price and date. For this first modelling approach, the predictors were intentionally limited to Classification and Date.

Technologies and Libraries

The project was developed using Python.

Main libraries:

* Python
* Pandas
* Scikit-learn
* NumPy
* Matplotlib

Data Preparation

The following preparation steps were performed before modelling:

1. The dataset was loaded and examined.
2. The relevant variables were selected.
3. The data was sorted chronologically.
4. The `Date` variable was converted into a numerical feature called `Days`.
5. `Classification`, which is categorical, was converted into numerical features using one-hot encoding.
6. The resulting dataset contained 1,415 observations and 26 model features.
7. The visual representation to understand the trend over time using line graph.

The 26 features consist of:

* 25 one-hot encoded classification features
* 1 numerical `Days` feature

Machine Learning Approach

The also project uses multiple linear regression.

Preprocessing

A `ColumnTransformer` was used to apply the appropriate preprocessing to the categorical variable while keeping the numerical date feature unchanged.

`OneHotEncoder` was used to convert product classifications into numerical binary features.

Train-Test Split

Because the dataset contains a time component, the data was split chronologically rather than randomly.

* Training set: 1,132 observations
* Testing set: 283 observations

The model is trained on earlier observations and evaluated on later observations.

This approach helps prevent information from future observations from being used to train the model.

Model Evaluation

Limitations

This project has several limitations:

* The available modelling data covers only May–December 2025, providing a relatively short historical period which limites use from make future predictions other than of 2025
* Product classifications have different numbers of observations.
* Some classifications have very few observations, which limits the amount of information available for learning their price behaviour.
* The model does not currently include other potentially important factors such as county, market, economic conditions, inflation, seasonality, or supply and demand variables.
* Predictions far beyond the observed period should therefore be interpreted cautiously.

Future Improvements

Future versions of the project could improve the model by incorporating:

* County and market information
* Inflation and economic indicators
* Agricultural production and supply data
* Seasonal variables
* Longer historical price records
* Additional machine learning algorithms
* Time-series forecasting methods
* Model comparison and hyperparameter tuning

Conclusion

This project demonstrates the application of a complete basic machine learning workflow to a real-world Kenyan pricing problem, from data preparation and feature engineering to regression modelling and evaluation.

The project also provides a foundation for exploring more advanced approaches to food price prediction and agricultural data science in Kenya.

Author

David Livingston

Bachelor of Data Science Student
Open University of Kenya

