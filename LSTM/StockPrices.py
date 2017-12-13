
# coding: utf-8

# # Predicting Stock Prices Using LSTM
# 
# Using keras for this one

# In[1]:


from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential


# In[2]:


import lstm, time


# In[3]:


# load data
X_train, y_train, X_test, y_test = lstm.load_data('data/sp500.csv', 50, True)


# In[4]:


# build model
model = Sequential()

model.add(LSTM(
    input_dim=1,
    output_dim=50,
    return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
    100,
    return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(
    output_dim=1))
model.add(Activation('linear'))

start = time.time()
model.compile(loss='mse', optimizer='rmsprop')
print('Compilation time : ', time.time() - start)


# In[5]:


print(model.summary())


# In[6]:


# Train the model
model.fit(
    X_train,
    y_train,
    batch_size=512,
    nb_epoch=1,
    validation_split=0.05)


# In[7]:


# Plot predictions
predictions = lstm.predict_sequences_multiple(model, X_test, 50, 50)
lstm.plot_results_multiple(predictions, y_test, 50)

