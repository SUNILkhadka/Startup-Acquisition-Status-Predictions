# Startup-Acquisition-Status-Prediction

### 1. Data Cleaning
    Dealing with unnecessary columns:
        a. Deleting columns that contains irrelevant and redundant information
        b. Deleting columns containing more than 98% of null values 
        c. Deleting duplicate values

    Dealing with missing values and outliers:
        a. Droping nan values of main columns 
        b. Removing outliers
        c. Checking for inconsistant data

### 2. DATA TRANSFORMATION
    Categorical data transformation:
    1. Changes in original data
        a. Converting object datatypes to pd.datetime and parsing years
        b. Generalize the categorical data using OneHotEncoder 

    2. Adding new columns
        a. Creating new features from exsisting columns and bringing insight to new values

    Numerical data transformation:
    1. Removing null values with their mean and most-frequent values depending on the columns  