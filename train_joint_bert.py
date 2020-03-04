from goo_format_reader import Reader
from src.JointBertModel import BERTVectorizer
from src.JointBertModel import TagsVectorizer
from src.JointBertModel import JointBertModel
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import pickle
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, cross_val_score
from plot_keras_history import plot_history
from tensorflow.keras.callbacks import ModelCheckpoint


# read command-line parameters
parser = argparse.ArgumentParser('Training the Joint BERT NLU model')
parser.add_argument('--train', '-t', help='Path to training data in Goo et al format', type=str, required=True)
parser.add_argument('--save', '-s', help='Folder path to save the trained model', type=str, required=True)
parser.add_argument('--epochs', '-e', help='Number of epochs', type=int, default=5, required=False)
parser.add_argument('--batch', '-bs', help='Batch size', type=int, default=64, required=False)

args = parser.parse_args()
data_folder_path = args.train
save_folder_path = args.save
epochs = args.epochs
batch_size = args.batch


tf.compat.v1.random.set_random_seed(7)
sess = tf.compat.v1.Session()


bert_model_hub_path = 'https://tfhub.dev/google/albert_base/1'

print('read data ...')
text_arr, tags_arr, intents = Reader.read(data_folder_path)

model = None

print('vectorize data ...')
bert_vectorizer = BERTVectorizer(sess, bert_model_hub_path)
input_ids, input_mask, segment_ids, valid_positions, sequence_lengths = bert_vectorizer.transform(text_arr)


print('vectorize tags ...')
tags_vectorizer = TagsVectorizer()
tags_vectorizer.fit(tags_arr)
tags = tags_vectorizer.transform(tags_arr, valid_positions)
slots_num = len(tags_vectorizer.label_encoder.classes_)


print('encode labels ...')
intents_label_encoder = LabelEncoder()
intents = intents_label_encoder.fit_transform(intents).astype(np.int32)
intents_num = len(intents_label_encoder.classes_)


if model is None:
    model = JointBertModel(slots_num, intents_num, bert_model_hub_path, sess, num_bert_fine_tune_layers=2)
else:
    model = JointBertModel.load(save_folder_path, sess)


print('training model ...')
X = np.concatenate((input_ids, input_mask, segment_ids, valid_positions, tags), axis=1)
Y = intents
split_width = input_ids.shape[1]

history = {}

for i in range(epochs):
    folds = StratifiedKFold(n_splits=5, shuffle=True).split(X, Y)

    for train_index, val_index in folds:
        X_train, X_val = X[train_index], X[val_index]
        Y_train, Y_val = Y[train_index], Y[val_index]

        Y_train = [X_train[:, 4*split_width:5*split_width], Y_train]
        X_train = [X_train[:, 0:split_width], X_train[:, split_width: 2*split_width],
                   X_train[:, 2*split_width: 3*split_width], X_train[:, 3*split_width: 4*split_width]]

        Y_val = [X_val[:, 4*split_width:5*split_width], Y_val]
        X_val = [X_val[:, 0:split_width], X_val[:, split_width: 2*split_width],
                 X_val[:, 2*split_width: 3*split_width], X_val[:, 3*split_width: 4*split_width]]

        X_train = (X_train[0], X_train[1], X_train[2], model.prepare_valid_positions(X_train[3]))
        X_val = (X_val[0], X_val[1], X_val[2], model.prepare_valid_positions(X_val[3]))

        hist = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=1, batch_size=batch_size)
        if history:
            history = {key: history[key] + hist.history[key] for key in hist.history}
        else:
            history = hist.history

plot_history(history)
plt.show()
plt.close()

print('Saving ..')
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)
    print('Folder `%s` created' % save_folder_path)
model.save(save_folder_path)
with open(os.path.join(save_folder_path, 'tags_vectorizer.pkl'), 'wb') as handle:
    pickle.dump(tags_vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(save_folder_path, 'intents_label_encoder.pkl'), 'wb') as handle:
    pickle.dump(intents_label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)


tf.compat.v1.reset_default_graph()
