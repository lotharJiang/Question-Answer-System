# Question-Answering-System
## File Structure
- **data**<br>A directory contains document data as well as training, devloping and testing dataset for the system, including training set for CNN question classifier.

- **classifier_model**<br>A directory contains checkpoint files for CNN question classifier.

- **answer_questions.py**<br>A python file aims to acquire answers of given question from given document.

- **vector_space_model.py**<br>A python file contains a class 'VectorSpaceModel'. It is utilized in the information retrieval phase in the system.

- **question_classifier.py**<br>A python file contains a class 'QuestionClassifier'. It is a convolutional neural network question classification model.

- **train_classifier.py**<br>A python file which is used to train the CNN question classifier.

- **utils.py**<br>A python file contains many userful tools in this system. E.g., text preprocessing, reading/writing csv file, etc.

## Usage
As the pretrained model of the CNN question classifier is not provided, the model needs to be trained before answering questions. It should take no more than an hour on a CPU device. Run the following command:
 - **python3 train_classifier.py**

Once the training finishes, the system is ready to run. It provides several options to do the information retrieval.

**Parameters**
- m: the method to use in information retrieval.
1. A variant of BM25
2. Cosine distance using tfidf
3. Cosine distance using tf

- b: BM25 parameter, format: k1,b,k3

For instance, if you want to do the information retrieval using BM25 with parameters k1 = 1.5, b = 0.5, k3 = 0, you should run the following command:
- **python3 answer_questions.py -m 1 -b 1.5,0.5,0**

When the execution ends, the results will be output to a CSV file under the same direcotry.
