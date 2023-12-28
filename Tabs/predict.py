import streamlit as st
from web_function import predict

def app(df, x, y):
  st.title("Prediksi Penghasilan")
  st.write("Masukkan data anda")
  
  col1, col2, col3 = st.columns(3)

  with col1:
    age = st.slider("Age", 0, 90, 17)
    hours_per_week = st.slider("Hours Per Week", 0, 99, 0)
    capitalGain = st.text_input("Capital Gain", 0, 6, 600)
  with col2:
    capitalLoss = st.text_input("Capital Loss", 0, 4, 100)
    marital_status = st.selectbox("Marital Status", ['Divorces', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'])
    occupation = st.selectbox("Occupation", ['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving'])
  with col3:
    relationship = st.selectbox("Relationship", ['Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife'])
    education = st.selectbox("Education", ['School', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Prof-school', 'Some-college'])
    workclass = st.selectbox("Workclass", ['Federal-gov', 'Local-gov', 'Never-worked', 'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'State-gov', 'Without-pay'])
  #convert marital_status, occupation, relationship, education and workclass to numerical data
  if marital_status == 'Divorces':
    marital_status = 0
  elif marital_status == 'Married-AF-spouse':
    marital_status = 1
  elif marital_status == 'Married-civ-spouse':
    marital_status = 2
  elif marital_status == 'Married-spouse-absent':
    marital_status = 3
  elif marital_status == 'Never-married':
    marital_status = 4
  elif marital_status == 'Separated':
    marital_status = 5
  elif marital_status == 'Widowed':
    marital_status = 6

  if occupation == 'Adm-clerical':
    occupation = 0
  elif occupation == 'Armed-Forces':
    occupation = 1
  elif occupation == 'Craft-repair':
    occupation = 2
  elif occupation == 'Exec-managerial':
    occupation = 3
  elif occupation == 'Farming-fishing':
    occupation = 4
  elif occupation == 'Handlers-cleaners':
    occupation = 5
  elif occupation == 'Machine-op-inspct':
    occupation = 6
  elif occupation == 'Other-service':
    occupation = 7
  elif occupation == 'Priv-house-serv':
    occupation = 8
  elif occupation == 'Prof-specialty':
    occupation = 9
  elif occupation == 'Protective-serv':
    occupation = 10
  elif occupation == 'Sales':
    occupation = 11
  elif occupation == 'Tech-support':
    occupation = 12
  elif occupation == 'Transport-moving':
    occupation = 13

  if relationship == 'Husband':
    relationship = 0
  elif relationship == 'Not-in-family':
    relationship = 1
  elif relationship == 'Other-relative':
    relationship = 2
  elif relationship == 'Own-child':
    relationship = 3
  elif relationship == 'Unmarried':
    relationship = 4
  elif relationship == 'Wife':
    relationship = 5

  if education == 'School':
    education = 0
  elif education == 'Assoc-acdm':
    education = 1
  elif education == 'Assoc-voc':
    education = 2
  elif education == 'Bachelors':
    education = 3
  elif education == 'Doctorate':
    education = 4
  elif education == 'HS-grad':
    education = 5
  elif education == 'Masters':
    education = 6
  elif education == 'Prof-school':
    education = 7
  elif education == 'Some-college':
    education = 8

  if workclass == 'Federal-gov':
    workclass = 0
  elif workclass == 'Local-gov':
    workclass = 1
  elif workclass == 'Never-worked':
    workclass = 2
  elif workclass == 'Private':
    workclass = 3
  elif workclass == 'Self-emp-not-inc':
    workclass = 4
  elif workclass == 'Self-emp-inc':
    workclass = 5
  elif workclass == 'State-gov':
    workclass = 6
  elif workclass == 'Without-pay':
    workclass = 7
  
  features = [age, workclass, education, marital_status, occupation, relationship, capitalGain, capitalLoss, hours_per_week]

  if st.button("Predict"):
    pred, score = predict(x, y, features)

    if pred == 0:
      st.warning("Penghasilan anda kurang dari 50K")
    else:
      st.success("Penghasilan anda lebih dari 50K")

    st.write("Akurasi: ", (score*100), "%")