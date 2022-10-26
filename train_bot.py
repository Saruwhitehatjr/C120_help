#Model Training Lib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.optimizers import Adam

from data_preprocessing import preprocess_train_data

def train_bot_model(train_x,train_y):
    model=Sequential([
        Dense(128,input_shape=(len(train_x[0]),),activation='relu'),
        Dropout(0.5),
        Dense(64,activation='relu'),
        Dropout(0.5),
        Dense(len(train_y[0]),activation='softmax')
    ])

    #Compile model
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    #Fit and save model
    history=model.fit(train_x,train_y,epochs=200,batch_size=5, verbose=True)
    model.save('chatbot_model.h5',history)
    print("Model file created and saved")

# Call methods
train_x,train_y=preprocess_train_data()
train_bot_model(train_x,train_y)