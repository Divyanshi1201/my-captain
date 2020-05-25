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



#load data
#loading data and opening our input datain the form of txt
#project Gutenberg is where the data can be found
file=open(r"C:\Users\Kishore\Documents\frankenstein_2.txt").read()

#tokenization
#standardisation
#tokenization is the process of breaking a stream of text up into words phreses or symbolsor a meaningful elements
def tokenize_words(input):
    #lowecase everything to standardise it
    input=input.lower()
    #instantiating the tokenizer
    tokenizer=RegexpTokenizer(r"\w+")
    #tokenzing the text into tokens
    tokens=tokenizer.tokenize(input)
    #filtering the stopwords using lambda
    filtered=filter(lambda token:token not in stopwords.words("english"),tokens)
    return"".join(filtered)
#preprocess the input data
processed_inputs=tokenize_words(file)


#characters to numbers
#convert characters in our input to numbers
#we will sort out the list of all characters that appearin our i/p text and then use the enumerate function to get numbers that
#represent characters
#we will then create a dictionary that stores keys and values or the characters and numbers that represent them
chars=sorted(list(set(processed_inputs)))
char_to_num=dict((c,i)for i,c in enumerate(chars))

#checking if char to num has worked
#just so we get an idea of whether our process of converting words to characters has worked
#we print the length of our variables
input_len=len(processed_inputs)
vocab_len=len(chars)
print("total number of characters:", input_len)
print("total vocab:",vocab_len)



#seq length
#we are defining how long we want our individual sequence here
#an individual sequence is a complete mapping of input characters as integers
seq_length=100
x_data=[]
y_data=[]


#loop through the sequences
#here we are goingthrough the entire list ofi/ps and converting the chars to numbers with a for loop
#this will create a bunch of sequence where each sequence starts with the next character in thei/p data
#begining with the first character
for i in range(0,input_len-seq_length,1):
    #define i/p and o/p sequence
    #i/p is current character plus desired sequence length
    in_seq=processed_inputs[i:i + seq_length]
    #out sequence is the initial character plus total sequecne length
    out_seq=processed_inputs[i + seq_length]
    #converting the list of characters to integers based on previous values and appending the values to our lists
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])
#check and see how many total input sequence we have
    n_patterns=len(x_data)
print("total patterns:",n_patterns)


#convert input sequence into np array and so that our network can use
X=numpy.reshape(x_data,(n_patterns,seq_length,1))
X=X/float(vocab_len)

#one-hot encoding our label data
y=np_utils.to_categorical(y_data)

#creating the model
#creating a sequential model
#dropout is used to avoid overfitting
model = Sequential()
model.add(LSTM(256,input_shape=(X.shape[1],X.shape[2]),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1],activation="softmax"))

#complile the model
model.compile(loss="categorical_crossentropy",optimizer="adam")

#saving weights
filepath="model_weights_saved.hdf5"
checkpoint=ModelCheckpoint(filepath,monitor="loss",verbose=1,save_best_only=True, mode="min")
desired_callbacks=[checkpoint]

#fit the model and let it train
model.fit(X,y,epochs=4,batch_size=256,callbacks=desired_callbacks)

#recompile the model with saved weights
filename="model_weights_saved.hdf5"
model.load_weights(filename)
model.compile(loss="categorical_crossentropy",optimizer="adam")

#output of the model back into characters
num_to_char=dict((i,c) for i,c in enumerate(chars))

#random seed to help generate
start=numpy.random.randint(0,len(x_data)-1)
pattern=x_data(start)
print("random seed: ")
print("\"",''.join((num_to_char(value) for value in patterns)),"\"")
