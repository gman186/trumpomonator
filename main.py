pathToControl = "Data/control.csv"
pathToTrump = "Data/trump.csv"
pathToPolitical = "Data/political.csv"
pathToBiden = "Data/biden.csv"
import csv
import tensorflow as tf
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.layers import Embedding

from tensorflow.keras.layers import Flatten


from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
opt = Adam(learning_rate = 0.003) #define optimizer
with open(pathToControl) as file:
    controlTexts = []
    info = csv.reader(file)
    for row in info:
        controlTexts.append(row[1])

with open(pathToTrump) as file:
    trumpTexts = []
    info = csv.reader(file)
    for row in info:
        trumpTexts.append(row[1])
    trumpTexts = trumpTexts[1:]

with open(pathToPolitical) as file:
    politicalTexts = []
    info = csv.reader(file)
    for row in info:
        politicalTexts.append(row[2])
    politicalTexts = politicalTexts[1:]

with open(pathToBiden) as file:
    bidenTexts = []
    info = csv.reader(file)
    for row in info:
        bidenTexts.append(row[0])
    bidenTexts = bidenTexts[1:]


front = TextVectorization(split="whitespace", output_mode="multi_hot", standardize=None) #define first layer
front.adapt(trumpTexts)
model = Sequential([ #create and define model
    front, #use text vectorize layer as first layer
    #Embedding(front.vocabulary_size(),16),
    #Flatten(), 
    Dense(128, activation="relu"),
    Dropout(rate=0.5),
    Dense(128,activation="relu"),
    Dropout(rate=0.5),
    Dense(256,activation="relu"),
    Dropout(rate=0.5),
    Dense(128,activation="relu"),
    Dropout(rate=0.5),
    Dense(64,activation="relu"),
    Dense(16,activation="relu"),
    Dense(1,activation="sigmoid")
])
model.compile(loss="mae")


def prepData(political,control, biden, trump): #sort, combine, and label the datasets
    Trump = []
    Control = []
    Political = []
    Biden = []
    for controlTweet in control:
        Control.append((controlTweet,0)) #label control data as not trump
    for politicalTweet in political:
        Political.append((politicalTweet,0)) #label politician data as not trump
    for bidenTweet in biden:
        Biden.append((bidenTweet,0)) #label biden's data as not trump
    for trumpTweet in trump:
        Trump.append((trumpTweet,1)) #label trump's data as trump
    random.shuffle(Political) #shuffle republican and democrat tweets
    random.shuffle(Biden) #shuffle republican and democrat tweets
    random.shuffle(Trump) #shuffle republican and democrat tweets
    random.shuffle(Control) #shuffle republican and democrat tweets

    combinedData =  Trump[:5000] + Control[:5000] + Political[:5000] + Biden[:5000] #add 5000 samples of each dataset
    inp = []
    out = []
    for item in combinedData: #format them into lists of lists
        inp.append([item[0]])
        out.append([item[1]]) 
    return inp,out




inp,out = prepData(politicalTexts,controlTexts,bidenTexts,trumpTexts) #prepare the lists of texts to be labeled
model.fit(tf.convert_to_tensor(inp),tf.convert_to_tensor(out),epochs=10,batch_size=32)

while True:
    print("Type tweet:")
    tweet= input("")
    probability = 100*model.predict(tf.convert_to_tensor([tweet]))[0][0] #calculate probability if it was trump
    if probability >=50:
        print("We are " + str(probability)+"% sure this was Trump") #print probability if it was likely trump
    else:
        print("We are " + str(100-probability)+"% sure this was not Trump") #print probability if it was likely not trump