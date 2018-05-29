import tensorflow as tf
import numpy as np
import os
import utils
from sklearn.utils import shuffle
from question_classifier import QuestionClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

# Get training set, vocabulary and question classes
x_train, y_train, vocabulary, classes = utils.get_training_dataset_info()

# Shuffle dataset
x_shuffled,y_shuffled = shuffle(x_train,y_train)

# Split dataset
x_train,x_dev,y_train,y_dev = train_test_split(x_shuffled, y_shuffled, test_size=0.1)

# Create directory
checkpoint_dir = 'classifier_model/checkpoints/'
checkpoint_prefix = checkpoint_dir + "classifier_model"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

tf.reset_default_graph()
sess = tf.Session()
with sess.as_default():

    classifier = QuestionClassifier(
        input_size=x_train.shape[1],
        n_classes=len(classes),
        vocabulary_size=len(vocabulary))

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
    init = tf.global_variables_initializer()
    sess.run(init)

    loss_dev = []
    accuracy_dev = []
    precision_dev = []
    recall_dev = []
    f1score_dev = []
    step = 0

    # Get training batches
    batches = utils.batch_generater(
        list(zip(x_train, y_train)), batch_size=128, num_epochs=10)

    for batch in batches:
        x_batch, y_batch = zip(*batch)

        # Training
        _ = sess.run(classifier.train_op, feed_dict={classifier.question: x_batch, classifier.label: y_batch, classifier.dropout_rate: 0.5})
        step += 1

        # Evaluate the classifier_model every 100 steps
        if step % 100 == 0:

            y_true = np.argmax(y_dev, 1)

            # Get labels, loss and accuracy
            y_predict, loss, accuracy = sess.run(
                [classifier.predictions, classifier.loss, classifier.accuracy],
                feed_dict={classifier.question: x_dev, classifier.label: y_dev, classifier.dropout_rate: 1.0})

            # Calculate precision, recall, f1-score
            precision, recall, f1score, _ = precision_recall_fscore_support(y_true, y_predict, average='macro')

            print("\nEvaluation:")
            print("Step:\t"+str(step))
            print("Loss:\t"+str(loss))
            print("Accuracy:\t"+str(accuracy))
            print("Precision:\t"+str(precision))
            print("Recall:\t"+str(recall))
            print("F1-score:\t"+str(f1score))

            loss_dev.append(loss)
            accuracy_dev.append(accuracy)
            precision_dev.append(precision)
            recall_dev.append(recall)
            f1score_dev.append(f1score)

        # Save the classifier_model every 100 steps
        if step % 100 == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=step)

# Save the evaluating results
np.save('loss_dev.npy', np.array(loss_dev))
np.save('accuracy_dev.npy', np.array(accuracy_dev))
np.save('precision_dev.npy', np.array(precision_dev))
np.save('recall_dev.npy', np.array(recall_dev))
np.save('f1score_dev.npy', np.array(f1score_dev))