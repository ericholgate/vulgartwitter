"""
Bi-Directional LSTM

Pass the cleaned version of the data (see clean_data.py) into the model.

Required parameters:
--train=<path to training data set>
--validation_data=<path to validation data set> 
--initial_embed_weights=<path to initial embedding weights>
--prefix=<prefix to save model>

Optional parameters:
--rnndim=<rnn dimension, default=128>
--dropout=<dropout rate, default=0.4>
--maxsentlen=<maximum length of tweets by number of words, default=60>
--num_cat=<number of categories, default=5>
--lr=<learning rate, default=0.001>
--only_testing=<boolean if you only want to load a saved model, default=False>

--concat=<boolean if using concat method, default=False>
--insert=<boolean if using insert method, default=False>
--mask=<boolean if using mask method, default=False>

Example usage:
python3 bilstm.py --train=<path> --test=<path> --prefix=example --concat=True

Returns:
- Saves model as h5 and json files to ./training
- Saves model at checkpoints to ./training/<prefix>_{epoch:02d}-{loss:.4f}.hdf5
- Prints summary of model
If a test set is provided
- Saves predictions of test set to ./training/predictions
- Prints micro mean absolute error
- Prints per class mean absolute error
"""


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, concatenate
from keras.utils.np_utils import to_categorical
from keras.engine import Input
from keras.models import Model
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import mean_absolute_error

import pprint
import logging
import numpy as np
import argparse
import pandas as pd
import csv


class BILSTM:
    def __init__(self, param):
        self.params = param

        self.tokenizer = Tokenizer(split=" ")

        # Initialize validation sentences
        self.xval, self.y_cat_val, self.yval, self.num_vulgar_val = self.clean_data(self.params["validation_data"])
        self.init_validation_sents()
        if self.params["concat"]:
            self.validation_data = (self.allxval, self.y_cat_val)
        else:
            self.validation_data = (self.xval, self.y_cat_val)

        if self.params['test'] is not None:
            # Initialize test sentences
            self.xtest, self.y_cat_test, self.ytest, self.num_vulgar_test = self.clean_data(self.params["test"])
            self.init_test_sents()

        # Initialize training sentences
        self.xtrain, self.y_cat_train, self.ytrain, self.num_vulgar = self.clean_data(self.params["train"])
        self.init_train_sents()

        # Initialize model
        self.model = self.init_bilstm()

    # Reads in data and returns x, y, y-categorical, and num_vulgar values
    def clean_data(self, f_name):
        df = pd.read_csv(f_name, sep="\t")

        # Which format of tweet to use?
        if self.params['insert']:
            col = 'insert_tweet'
        elif self.params['mask']:
            col = 'masked_tweet'
        else:
            col = 'Tweet'

        x = np.asarray(df[col])
        y = np.asarray(df["Majority"])
        y_cat = to_categorical(y - 1, num_classes=params['num_cat'])
        y_cat = np.asarray(y_cat)
        num_vulgar = np.asarray(df['num_vulgar'])

        return x, y_cat, y,  num_vulgar

    # Tokenizes and pads training sentences
    def init_train_sents(self):
        self.tokenizer.fit_on_texts(self.xtrain)
        sequences = self.tokenizer.texts_to_sequences(self.xtrain)
        logging.info('Found %s unique tokens.' % len(self.tokenizer.word_index))
        self.xtrain = pad_sequences(sequences, maxlen=self.params["maxsentlen"])
        if self.params["concat"]:
            self.allxtrain = [self.xtrain, self.num_vulgar]
            logging.debug("Num vulgar shape: {}".format(self.num_vulgar.shape))

        logging.info('Shape of X: {0}'.format(self.xtrain.shape))
        logging.info('Shape of Y: {0}'.format(self.y_cat_train.shape))
        logging.info("Train data init complete")

    # Tokenizes and pads test sentences
    def init_test_sents(self):
        self.tokenizer.fit_on_texts(self.xtest)
        sequences = self.tokenizer.texts_to_sequences(self.xtest)
        self.xtest = pad_sequences(sequences, maxlen=self.params["maxsentlen"])
        if self.params["concat"]:
            self.allxtest = [self.xtest, self.num_vulgar_test]

        logging.info("Test data init complete")

    # Tokenizes and pads validation sentences
    def init_validation_sents(self):
        self.tokenizer.fit_on_texts(self.xval)
        sequences = self.tokenizer.texts_to_sequences(self.xval)
        self.xval = pad_sequences(sequences, maxlen=self.params["maxsentlen"])
        if self.params["concat"]:
            self.allxval = [self.xval, self.num_vulgar_val]

        logging.info("Validation data init complete")

    # Called by init_bilstm method
    # Reads in embedding weights and return embedding layer
    def _init_embeddings(self):
        embeddings_index = {}

        with open(self.params['initial_embed_weights']) as f:
            embedding_info = f.readline().strip().split()
            num_words, embed_dim = eval(embedding_info[0]), eval(embedding_info[1])
            logging.info("Number of word vectors: {}".format(num_words))
            logging.info("Embedding Dimension: {}".format(embed_dim))

            for line in f:
                values = line.strip().split()
                if values == []:
                    pass
                else:
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs / np.linalg.norm(coefs)
            f.close()
        logging.debug("Embed dim: {}".format(embed_dim))
        embedding_matrix = np.zeros((len(self.tokenizer.word_index) + 1, embed_dim))
        unk_dict = {}
        for word, i in self.tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            elif word in unk_dict:
                embedding_matrix[i] = unk_dict[word]
            else:
                # random init, see https://github.com/bwallace/CNN-for-text-classification/blob/master/CNN_text.py
                unk_embed = np.random.random(embed_dim) * -2 + 1
                unk_dict[word] = unk_embed
                embedding_matrix[i] = unk_dict[word]
        embedding_layer = Embedding(input_dim=len(self.tokenizer.word_index) + 1,
                                    output_dim=embed_dim,
                                    weights=[embedding_matrix],
                                    input_length=self.params["maxsentlen"],
                                    mask_zero=True)
        logging.info("Embedding layer completed")
        logging.info(str(len(unk_dict)) + " unknown words")
        return embedding_layer

    # Builds model
    def init_bilstm(self):
        x_in = Input(shape=(self.xtrain.shape[1],), dtype='int32')
        embedding = self._init_embeddings()
        x_embed = embedding(x_in)
        bilstm = Bidirectional(LSTM(self.params['rnndim']))(x_embed)

        if self.params["concat"]:
            concat_in = Input(shape=(1,), dtype='float32')
            logging.debug("Concat in shape: {}".format(concat_in.shape))
            concat = concatenate([bilstm, concat_in], axis=-1)
            dense = Dense(self.params['num_cat'], activation="relu")(concat)
            drop_out = Dropout(self.params['dropout'])(dense)
            toplayer = Dense(params['num_cat'], activation="softmax")(drop_out)
            my_model = Model(inputs=[x_in, concat_in], outputs=[toplayer])
        else:
            dense = Dense(self.params['num_cat'], activation="relu")(bilstm)
            drop_out = Dropout(self.params['dropout'])(dense)
            toplayer = Dense(params['num_cat'], activation="softmax")(drop_out)
            my_model = Model(inputs=[x_in], outputs=[toplayer])

        my_model.layers[1].trainable = False
        adam = optimizers.Adam(lr=params["lr"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        my_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        my_model.summary()

        return my_model

    # Trains model
    def train(self):
        filepath = "./training/" + self.params['prefix']+ "_{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor="loss", verbose=1, save_best_only=True, mode="min")
        callbacks_list = [checkpoint, EarlyStopping(monitor="val_acc", patience=1)]

        if self.params['concat']:
            xtrain = self.allxtrain
            logging.debug("Self.allxtrain: {}".format(self.allxtrain))

        else:
            xtrain = self.xtrain

        self.model.fit(xtrain,
                       self.y_cat_train,
                       epochs=self.params['nepoch'],
                       batch_size=self.params['batch_size'],
                       verbose=True,
                       validation_data=self.validation_data,
                       callbacks=callbacks_list)

        # Save model as json and h5 file
        json_string = self.model.to_json()
        with open("./training/" + self.params['prefix'] + "_model.json", "w") as json_file:
            json_file.write(json_string)
        self.model.save_weights('./training/' + self.params['prefix'] + '_model_weights.h5')

    # Macro MAE
    def MMAE(self, pred):
        cats = set(self.ytest)
        mmae = 0
        cat_mmae = {}
        for cat in cats:
            class_sum = sum(abs(pred[i] - self.ytest[i]) for i in range(len(pred)) if self.ytest[i] == cat)
            class_mae = class_sum / (self.ytest == cat).sum()
            mmae += class_mae
            cat_mmae[cat] = class_mae

        return (mmae / len(cats)), cat_mmae

    # Loads saved model for testing
    def load(self, filename):
        self.model.load_weights(filename)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

    # Test model
    def test(self):
        save_preds = open('./training/predictions/' + self.params['prefix'] + '.csv', 'w')
        writer = csv.writer(save_preds, delimiter=',')
        writer.writerow(["Y_PRED", "Y_TRUE"])
        if self.params['concat']:
            preds = self.model.predict(self.allxtest)
        else:
            preds = self.model.predict(self.xtest)
        pred_int = np.array([np.argmax(x) + 1 for x in preds])

        for i in range(len(pred_int)):
            row = [pred_int[i], self.ytest[i]]
            writer.writerow(row)
        mae = mean_absolute_error(self.ytest, pred_int)
        mmae, cat_mmae = self.MMAE(pred_int)
        logging.info("MicroMAE: {}".format(mae))
        logging.info("CatMAE:")
        pprint.pprint(cat_mmae)
        save_preds.close()

if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--rnndim", help="Projection dimension for RNN",
                        type=int, default=128, action='store')
    parser.add_argument("--dropout", help="Dropout rate",
                        type=float, default=0.4, action='store')
    parser.add_argument("--validation_data", help="Validation data set",
                        type=str, default=None, action='store')
    parser.add_argument("--nepoch", type=int, default=10, action='store')
    parser.add_argument("--batch_size", type=int, default=256, action='store')
    parser.add_argument("--maxsentlen", type=int, default=60, action='store')
    parser.add_argument("--num_cat", type=int, default=5, action='store', help='Number of categories')
    parser.add_argument("--concat", type=bool, default=False, action='store', help='Concatonate num vulgar?')
    parser.add_argument("--insert", type=bool, default=False, action='store', help='Insert method?')
    parser.add_argument("--mask", type=bool, default=False, action='store', help='Mask method?')
    parser.add_argument("--lr", type=float, default=0.001, action='store', help='Learning rate')
    parser.add_argument("--initial_embed_weights", type=str,
                        help='File with initialized embeddings', action='store')
    parser.add_argument("--train", type=str, help='Training file', action='store')
    parser.add_argument("--test", type=str, default=None, help='Test file', action='store')
    parser.add_argument("--prefix", type=str, help='prefix for saving model', action='store')
    parser.add_argument("--only_testing", type=bool, default=False,
                        action='store', help="Boolean if you are loading a saved model")

    args = parser.parse_args()
    params = vars(args)

    if params["train"] is None:
        parser.error("Please specify a training set.")
    if params["validation_data"] is None:
        parser.error("Please specify a validation set.")
    if params["initial_embed_weights"] is None:
        parser.error("Please specify inital embedding weights.")
    if params["prefix"] is None:
        parser.error("Please specify a prefix to save your model.")
    try:
        pprint.pprint(params)
        model = BILSTM(params)
        if params['only_testing']:
            model.load('./training/' + params['prefix'] + '_model_weights.h5')
            model.test()
        else:
            model.train()
            if params["test"] is not None:
                model.load('./training/' + params['prefix'] + '_model_weights.h5')
                model.test()
    except KeyboardInterrupt:
        logging.warning("Keyboard Interrupt.")
