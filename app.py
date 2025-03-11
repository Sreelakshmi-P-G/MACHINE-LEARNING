import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv(r"C:\Users\laksh\Downloads\Crop damage.csv")

df['Number_Doses_Week'].fillna(df['Number_Doses_Week'].mean(),inplace=True)
df['Number_Weeks_Used'].fillna(df['Number_Weeks_Used'].mode()[0],inplace=True)
df['Number_Weeks_Quit'].fillna(df['Number_Weeks_Quit'].median(),inplace=True)

le1=LabelEncoder()
le2=LabelEncoder()
le3=LabelEncoder()
le4=LabelEncoder()
df['Crop_Type']=le1.fit_transform(df['Crop_Type'])
df['Soil_Type']=le2.fit_transform(df['Soil_Type'])
df['Pesticide_Use_Category']=le3.fit_transform(df['Pesticide_Use_Category'])
df['Season']=le4.fit_transform(df['Season'])

x=df.drop('Crop_Damage',axis=1)
y=df['Crop_Damage']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=42)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

model=GradientBoostingClassifier(n_estimators=150,random_state=42)
model.fit(x_train,y_train)

st.title("Crop Damage Prediction App")
st.header("Enter Required Crop Parameters")

i=st.text_input("Estimated Insects Count:",value="2000.0")
c=st.text_input("Crop Type(Rabi/Kharif):",value="Rabi")
c=le1.transform([c])
soil=st.text_input("Soil Type(Alluvial/Black-Cotton):",value="Alluvial")
soil=le2.transform([soil])
p=st.text_input("Pesticide Use Category(Insecticides/Bactericides/Herbicides):",value="Insecticides")
p=le3.transform([p])
d=st.text_input("Number of Doses per Week",value="20.0")
u=st.text_input("Number of Weeks Used:",value="25.0")
q=st.text_input("Number of Weeks Quit:",value="10.0")
s=st.text_input("Season(Summer/Monsoon/Winter):",value="Monsoon")
s=le4.transform([s])

if st.button("Predict"):
    try:
        input_data=[[
            float(i),
            float(c[0]),
            float(soil[0]),
            float(p[0]),
            float(d),
            float(u),
            float(q),
            float(s[0])
        ]]
        input_data = scaler.transform(input_data)
        prediction=model.predict(input_data)[0]

        st.subheader("Prediction Result")
        st.write(f"The predicted damage is *{prediction}*!")
    except ValueError:
        st.error("Please enter valid info for all fields.")
st.sidebar.title("About")
st.sidebar.info("This application uses a pre-trained Gradient Boost model to predict the extent"
    " of crop damage based on the specific parameters. Enter the feature values and click 'predict'.")