import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.utils import load_img, img_to_array
from keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

word_to_index = {}
with open ("C:/Users/Acer/Documents/Image Caption Generator/Image-Caption-Generator/Flickr30k_Dataset/dataset/word_to_index.pkl", 'rb') as file:
    word_to_index = pd.read_pickle(file, compression=None)

index_to_word = {}
with open ("C:/Users/Acer/Documents/Image Caption Generator/Image-Caption-Generator/Flickr30k_Dataset/dataset/index_to_word.pkl", 'rb') as file:
    index_to_word = pd.read_pickle(file, compression=None)

    
model = load_model('C:/Users/Acer/Documents/Image Caption Generator/Image-Caption-Generator/Flickr30k_Dataset/dataset/models/model_14.h5')

resnet50_model = ResNet50 (weights = 'imagenet', input_shape = (224, 224, 3))
resnet50_model = Model (resnet50_model.input, resnet50_model.layers[-2].output)
def predict_caption(photo):
    inp_text = "startseq"
    for i in range(80):
        sequence = [word_to_index[w] for w in inp_text.split() if w in word_to_index]
        sequence = pad_sequences([sequence], maxlen=80, padding='post')

        ypred = model.predict([photo, sequence])
        ypred = ypred.argmax()
        word = index_to_word[ypred]

        inp_text += (' ' + word)

        if word == 'endseq':
            break

    final_caption = inp_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption

def preprocess_image (img):
    img = load_img(img, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def encode_image (img):
    img = preprocess_image(img)
    feature_vector = resnet50_model.predict(img)
    # feature_vector = feature_vector.reshape((-1,))
    return feature_vector


def runModel(img_name):
    #img_name = input("enter the image name to generate:\t")
    photo = encode_image(img_name).reshape((1, 2048))
    caption = predict_caption(photo)
    print(caption)
    return caption

runModel("C:/Users/Acer/Documents/Image Caption Generator/Image-Caption-Generator/Flickr30k_Dataset/testimages/test20.jpg")