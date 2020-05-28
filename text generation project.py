#importing dependencies
import numpy
import nltk
nltk.download("stopwords")
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential 
from keras.layers import Dense, Dropout,LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


#load the data
#uploading the data to current working directory and then loading it
file=open("frankenstein.txt").read()

#tokenization
#tokenization is the process of convering the words to number vectorsas a result of which tokens representing the actual words
#are obtained
def tokenize_words(input):
    #converting the input in lowercase to standardise it
    input=input.lower()
    #initiating the tokenization
    tokenizer=RegexpTokenizer("r\w+")
    tokens=tokenizer.tokenize(input)
    filtered=filter(lambda token: token not in stopwords.words("english"),tokens)
    return "".join(filtered)

processed_inputs=tokenize_words(file)

#char to numbers
chars=sorted(list(set(processed_inputs)))
char_to_num=dict((c,i) for i,c in enumerate (chars))

#checking if char to num has worked
input_len=len(processed_inputs)
vocab_len=len(chars)
print("total numbers of characters:",input_len)
print("total vocab:",vocab_len)

#seq_length
#we are defining how long we want our individual sequence
#an individual sequence is complete mappinf of input characters as integers
seq_len=100
x_data=[]
y_data=[]


#loop through sequence
#this will create bunch of sequences where each sequence starts with thenext character in the input data.
#the sequence begining with the first character
for i in range(0,input_len - seq_len - 1):
    in_seq=processed_inputs[i:i+seq_len]
    out_seq=processed_inputs[i+seq_len]
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])
    
#total number of input sequence
n_patterns=len(x_data)
print("total patters:",n_patterns)

#convert input sequence to np array
x=numpy.reshape(x_data,(n_patterns,seq_len,1))
x=x/float(vocab_len)

#one hot encoding
from keras.utils import np_utils
y=np_utils.to_categorical(y_data)


#create the model
model=Sequential()
model.add(LSTM(254,input_shape=(x.shape[1],x.shape[2]),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1],activation="softmax"))


#compile the model
model.compile(loss='categorical_crossentropy',optimizer="adam")


#saving weights
filepath="model_weights_saved.hdf5"
checkpoint=ModelCheckpoint(filepath,monitor='loss',verbose=1,save_best_only=True,mode="min")
desired_callbacks=[checkpoint]


#fit the model and let it train
model.fit(x,y,epochs=4,batch_size=256,callbacks=desired_callbacks)

#recompile the model with saved weights
filename="model_weights_saved.hdf5"
model.load_weights(filename)
model.compile(loss="categorical_crossentropy",optimizer="adam")

#output of the model back into characters
num_to_char=dict((i,c) for i,c in enumerate(chars))

#random seed to help generate
start=numpy.random.randint(0,len(x_data)-1)
pattern=x_data[start]
print("random seed: ")
print("\"",''.join([num_to_char[value] for value in pattern]),"\"")


#generate the text
import sys
for i in range(1000):
    x=numpy.reshape(pattern,(1,len(pattern),1))
    x=x/float(vocab_len)
    prediction=model.predict(x,verbose=0)
    index=numpy.argmax(prediction)
    result=num_to_char[index]
    seq_in=[num_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern=pattern[1:len(pattern)]
    
    
