import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from create_sentiment_featuresets import create_feature_sets_and_labels

#train_x, train_y, test_x, test_y = create_feature_sets_and_labels('./files/pos.txt', './files/neg.txt')

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
batch_size = 100

#long_training = len(train_x[0])
long_training = 423

x = tf.placeholder('float', [None, long_training], name='x')
#x = tf.placeholder('float', name='x')
y = tf.placeholder('float', name='y')

hidden_1_layer = {'weights':tf.Variable(tf.random_normal([long_training, n_nodes_hl1])),
                  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'biases':tf.Variable(tf.random_normal([n_classes]))}

def neural_network_model(data):
    # y = (x * weights) + biases
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1) # f(x) = max(0, x)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output, hidden_1_layer['weights']

saver = tf.train.Saver()

def train_neural_network(x):
    prediction, whl1 = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    f = open('./outputs/loss.txt', 'w')
    f2 = open('./outputs/weights-epochs.csv', 'w')
    
    # feed forward + backpropagation = epoch
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0

            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size

            saver.save(sess, './outputs/model.ckpt')
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
            weights = sess.run(whl1[0])
            
            for weight in weights:
                f2.write(str(weight) + '\t')
            f2.write('\n')
            f.write(str(epoch_loss) + '\n')
        f.close()
        f2.close()

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))

    f2 = open('./outputs/weights-epochs.csv', 'r')
    f3 = open('./outputs/weights-epochs2.csv', 'w')
    lines = f2.readlines()
    line_index = 0
    len_columns = len(lines[0].split('\t'))
    for j in range(0, len_columns):
        for line in lines:
            columns = line.split('\t')
            f3.write(str(columns[j]) + '\t')
        f3.write('\n')

    f3.close()

def use_neural_network(input_data):
    prediction, whl1 = neural_network_model(x)
    lemmatizer = WordNetLemmatizer()
    with open('./files/lexicon.pickle','rb') as f:
        lexicon = pickle.load(f)
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./outputs/model.ckpt")
        current_words = word_tokenize(input_data.lower())
        current_words = [lemmatizer.lemmatize(i) for i in current_words]
        features = np.zeros(len(lexicon))

        for word in current_words:
            if word.lower() in lexicon:
                index_value = lexicon.index(word.lower())

                features[index_value] += 1

        features = np.array(list(features))
        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}),1)))
        if result[0] == 0:
            print('Positive:', input_data)
        elif result[0] == 1:
            print('Negative:', input_data)

#train_neural_network(x)

print('Bad predictions')
use_neural_network("Dollar is the worst currency in the world")
use_neural_network("This was the best store i've ever seen")

print('Good predictions')
use_neural_network("Messi is the best")
use_neural_network("The national parlament is a joke")

#use_neural_network("My father is the best in the world")
inp = input('Test your own phrase:')
use_neural_network(inp)








