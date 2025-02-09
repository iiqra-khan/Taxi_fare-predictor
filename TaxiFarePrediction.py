import streamlit as st;
import pandas as pd;
import numpy as np;
import seaborn as sns;
import matplotlib.pyplot as plt
from sklearn.preprocessing import (MinMaxScaler, QuantileTransformer)
from sklearn.model_selection import (train_test_split, cross_val_predict,RepeatedStratifiedKFold)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.title('Taxi Fare Prediction App')

df = pd.read_csv(r'C:\Users\iqra khan\OneDrive\Desktop\MSC\MachineLearning\Streamlit\Code\TaxiFarePredictor\taxi_trip_pricing.csv')

df = pd.get_dummies(df, drop_first=False, dtype=int)

# df = df.drop(columns=[])
for column in df.columns:
    mode_val = df[column].mode()[0]
    df[column] = df.loc[:, column].fillna(mode_val)

with st.expander("DataFrame"):
    st.dataframe(df, hide_index=True)

with st.expander("Heatmap"):
    f, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, mask = np.zeros_like(corr, dtype = bool), 
                cmap = sns.diverging_palette(240,10,as_cmap = True),
            square = True, ax = ax)
    st.pyplot(f)

qt = QuantileTransformer(n_quantiles=10, output_distribution='uniform')
transformed_df = qt.fit_transform(df)

# x = df.drop(['Trip_Price'])
transformed_df = pd.DataFrame(transformed_df, columns=df.columns)


y = transformed_df['Trip_Price']
x = transformed_df.loc[:, transformed_df.columns!= 'Trip_Price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)
lr = LinearRegression()
lr.fit(x_train, y_train)

def predict(z):
    z = pd.get_dummies(z, drop_first=False, dtype=int)
    for column in x.columns:
        if column not in z.columns:
            z[column] = 0
    cols = list(x.columns)
    z = z[cols]
    predict = lr.predict(z)
    return predict[0]

st.title('Linear Regression Model ')

with st.form('Trip Details'):
    st.write("Enter Details of your trip ")

    Trip_Distance_km = st.number_input("Trip Distance(km)", min_value=0.1, max_value=100.0, value = 1.0)

    Passenger_Count = st.selectbox(
        'Total Passengers',
        (1, 2, 3, 4)
    )
    Base_Fare = st.number_input("Base Fare ($)", min_value=0.0, max_value=10.0, value=2.5)

    Per_Km_Rate = st.number_input('Per Km Rate ($)', min_value=1.0, max_value=2.0, value=1.0)

    Per_Minute_Rate = st.number_input('Per Minute Rate ($)', min_value=0.1, max_value=0.5, value=0.1)

    Trip_Duration_Minutes = st.number_input("Trip Duration (Min)", min_value=5.0, max_value=120.0, value=5.0)

    Time_of_Day = st.selectbox(
        'Time Of Day', 
        ('Morning', 'Afternoon', 'Evening', 'Night')
    )

    Traffic_Conditions = st.selectbox(
        'Traffic Condition',
        ('High', 'Low', 'Medium')
    )

    Weather = st.selectbox(
        'Weather Condition',
        ('Clear', 'Rain', 'Snow')
    )

    Day_of_Week = st.selectbox(
        "Day of Week",
        ("Weekday", "Weekend")
    )

    submitted = st.form_submit_button("Predict Fare")

    if submitted:
        input_data = pd.DataFrame({
            'Trip_Distance_km': [Trip_Distance_km],
            'Passenger_Count': [Passenger_Count],
            'Base_Fare': [Base_Fare],
            "Per_Km_Rate": [Per_Km_Rate],
            "Per_Minute_Rate": [Per_Minute_Rate],
            "Trip_Duration_Minutes": [Trip_Duration_Minutes],
            "Time_of_Day":[Time_of_Day],
            "Traffic_Conditions":[Traffic_Conditions],
            "Weather": [Weather],
            "Day_of_Week":[Day_of_Week]
        })
        predicted_fare = predict(input_data) 
        st.success(f"The predicted fare is: ${predicted_fare:.2f}")

# CHANGES TO BE MADE
# do cv deal with scaling etc


