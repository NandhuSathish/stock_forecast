import math
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import date
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import streamlit as st
plt.style.use('fivethirtyeight')

start='2012-01-01'
end=date.today().strftime("%Y-%m-%d")

st.title('Stock Trend Prediction')
form = st.form(key='my-form')
name = form.text_input('Enter Stock Ticker')
submit = form.form_submit_button('Submit')

# st.write('Press submit to have your name printed below')

# if submit:
#     st.write(f'hello {name}')


@st.cache(allow_output_mutation=True)
def get_data():
    return []

user_id = name
if submit:
    get_data().append({"Stocks": user_id})

st.subheader('Previous searched stocks')
st.write(pd.DataFrame(get_data()))

# st.title('Stock Trend Prediction')

# user_input = st.text_input('Enter Stock Ticker','TTM')
user_input = name
tickerData = yf.Ticker(user_input)

data_load_state = st.text('Loading data...')
df = tickerData.history(period='1d', start=start, end=end)
data_load_state.text(user_input + '  data Loading ... done!')



#Describe Data
st.subheader('Data from 2012-01-01  -  '+end)
st.write(df.describe())

#Visualizations
st.subheader('Closing Price vs Time Chart')
    # fig = plt.figure(figsize = (12,6))
    # plt.plot(df.Close)
    # st.pyplot(fig)
st.line_chart(df.Close)


st.subheader('Closing Price vs 100 Days MA')
ma100=df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.plot(ma100)
st.pyplot(fig)

st.subheader('Closing Price vs 100 & 200 Days MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close,'b')
plt.plot(ma100)
plt.plot(ma200)
st.pyplot(fig)

#divide the data into training and testing data
data = df.filter(['Close'])
dataset = data.values
training_data_len = math.ceil( len(dataset)*.8)

scalar = MinMaxScaler(feature_range=(0,1))
scaled_data = scalar.fit_transform(dataset)

training_data = scaled_data[0:training_data_len,:]
x_train = []
y_train = []
for i in range(60,len(training_data)):
  x_train.append(training_data[i-60:i,0])
  y_train.append(training_data[i,0])

x_train,y_train = np.array(x_train),np.array(y_train)
x_train=np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


#Creating the modal

# model=Sequential()
# model.add(LSTM(50,return_sequences=True, input_shape=(x_train.shape[1],1)))
# model.add(LSTM(50,return_sequences=False))
# model.add(Dense(25))
# model.add(Dense(1))
# model.compile(optimizer='adam',loss='mean_squared_error')
#model.fit(x_train,y_train,batch_size=1,epochs=6)
@st.cache(allow_output_mutation=True)
model = load_model('keras_model.h5')

#Testing the modal on data
test_data=scaled_data[training_data_len-60:,:]
x_test=[]
y_test=dataset[training_data_len:,:]
for i in range(60,len(test_data)):
  x_test.append(test_data[i-60:i,0])

x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

predictions=model.predict(x_test)
predictions=scalar.inverse_transform(predictions)
rmse = np.sqrt(np.mean(np.power((np.array(y_test)-np.array(predictions)),2)))

train=data[:training_data_len]
valid=data[training_data_len:]
valid['predictions']=predictions


st.subheader('Model Prediction Graph')
fig=plt.figure(figsize=(16,8))
plt.title('model')
plt.xlabel('date',fontsize=18)
plt.ylabel('Close price USD($)',fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close','predictions']])
plt.legend(['Train','val','predictions'], loc='lower right')
st.pyplot(fig)


st.subheader('Close price VS Prediction')
st.write(valid.tail())
