Implementation of Logistic regression algorithm

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df = pd.read_csv('data.csv')
df.head()
X = df.drop(["Flood Risk","Name"], axis=1)
X = pd.get_dummies(X)
y = df["Flood Risk"]
X.head()
y.head()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=42)
model = linear_model.LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy_score(y_test,y_pred)

Building an optimized simple Neural Network

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
# Load the dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# Normalize pixel values to the range [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0
# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
# Build the model
model = Sequential([
Flatten(input_shape=(28, 28)), # Flatten 28x28 input images
Dense(128, activation='relu'), # First hidden layer with 128 units
Dense(64, activation='relu'), # Second hidden layer with 64 units
Dense(10, activation='softmax') # Output layer with 10 units (one per class)
])
# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
loss='categorical_crossentropy',
metrics=['accuracy'])
# Print the model summary
model.summary()
# Train the model
history = model.fit(X_train, y_train,
validation_data=(X_test, y_test),
epochs=10, batch_size=32)
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")
# Predict classes on the test set
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)
y_true_classes = y_test.argmax(axis=1)
# Print classification report
print("Classification Report:")
print(classification_report(y_true_classes, y_pred_classes))
import matplotlib.pyplot as plt
# Plot training & validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# Plot training & validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

Implementation of Perceptron

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Load Iris dataset
iris = load_iris()
X = iris.data[:100] # Select only the first 100 samples (setosa and versicolor)
y = iris.target[:100] # Select the corresponding labels (0 or 1)
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Standardize the features for better convergence
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
class Perceptron:
  def __init__(self, learning_rate=0.01, n_iters=1000):
    self.lr = learning_rate
    self.n_iters = n_iters
    self.weights = None
    self.bias = None


  def fit(self, X, y):
    n_samples, n_features = X.shape
    # Initialize weights and bias
    self.weights = np.zeros(n_features)
    self.bias = 0
    # Convert labels to -1 and 1 (Perceptron requires this)
    y_ = np.where(y <= 0, -1, 1)
    # Gradient descent
    for _ in range(self.n_iters):
      for idx, x_i in enumerate(X):
        linear_output = np.dot(x_i, self.weights) + self.bias
        y_pred = np.sign(linear_output)
        # Update weights and bias if the prediction is wrong
        if y_[idx] != y_pred:
          self.weights += self.lr * y_[idx] * x_i
          self.bias += self.lr * y_[idx]
         
  # The predict method should be outside the fit method
  def predict(self, X):  
    linear_output = np.dot(X, self.weights) + self.bias
    return np.where(linear_output >= 0, 1, 0)
# Create a Perceptron instance and train it
perceptron = Perceptron(learning_rate=0.01, n_iters=1000)
perceptron.fit(X_train, y_train)
# Predict on test data
y_pred = perceptron.predict(X_test)
# Print predictions
print(f"Predictions: {y_pred}")
print(f"Actual labels: {y_test}")

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

Program based on Multilayer perceptron

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
#load iris dataset
iris = load_iris()
x = iris.data
y = iris.target


#one hot encode the target variable
y = to_categorical(y)


#split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


#standadize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# initalize the perceptron model
perceptron_model = Sequential()


#add the input layer with 4 neurons (one for each feature) and the output layer with 3 neurons
perceptron_model.add(Dense(3, activation='softmax', input_dim=4))


#compile the model
perceptron_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


#train the model
perceptron_model.fit(x_train, y_train, epochs=50, batch_size=5, verbose=1)


#evaluate the model on the test set
loss, accuracy = perceptron_model.evaluate(x_test, y_test)

#evalute the model
loss, accuracy = perceptron_model.evaluate(x_test, y_test)
print(f'preceptron model accuracy: {accuracy * 100:.2f}%')

#intialize the multi -layer perceptron model
mlp_model = Sequential()


#add the input layer with 4 neurons (one for each feature) and the output layer with 3 neurons
mlp_model.add(Dense(8, activation='relu', input_dim=4))
mlp_model.add(Dense(3, activation='softmax'))
mlp_model.add(Dense(2, activation='softmax'))
mlp_model.add(Dense(3, activation='softmax'))




#compile the model
mlp_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


#train the model
mlp_model.fit(x_train, y_train, epochs=50, batch_size=5, verbose=1)



#evaluate the model on the test set
loss, accuracy = mlp_model.evaluate(x_test, y_test)
print(f'preceptron model accuracy: {accuracy * 100:.2f}%')


 Introduction to keras library
 
 import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
data = pd.read_csv(url, header=None)


# Split features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Build the ANN model
model = Sequential()
model.add(Dense(8, input_shape=(X_train.shape[1],), activation='relu'))  # Input layer
model.add(Dense(8, activation='relu'))  # Hidden layer
model.add(Dense(1, activation='sigmoid'))  # Output layer


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Summary of the model
model.summary()


# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_test, y_test))


# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

Implementation of Artificial Neural Network

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
import tensorflow as tf


# Load the dataset
data = pd.read_csv("/content/winequality-red - winequality-red.csv")


# Preprocess the data
X = data.drop('quality', axis=1)
y = data['quality']


# One-hot encode the quality labels for multi-class classification
y_encoded = to_categorical(y - 1)  # Subtract 1 if labels are 1-indexed


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)


# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Build the model with more hidden layers
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(90, activation='relu'),
    Dense(90, activation='relu'),
    Dense(80, activation='relu'),
    Dense(y_encoded.shape[1], activation='softmax')  # Number of classes
])


# Compile the model
model.compile(optimizer='adam', loss='squared_hinge', metrics=['accuracy'])
# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=1)


# Print the model summary
model.summary()


# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")


# Save the model architecture to a file
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)


# If you are in a Jupyter notebook, you can display the image directly
from IPython.display import Image
Image(filename='model_architecture.png', width=500, height=500)


ANN

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#importing the dataset
dataset=pd.read_csv('/content/Indian Liver Patient Dataset (ILPD).csv')
#checking the head of our dataset
dataset.head()

#checking the  info of the dataset
dataset.info()

dataset.shape

#Counting the ocurrence of values of Dataset column
dataset['Dataset'].value_counts()

#Converting the Binary value of the Dataset column i.e 1 and 2
#to 0 and 1 as our ANN model got confused and the loss function become negative
dataset.loc[dataset['Dataset']==1,'Dataset']=0
dataset.loc[dataset['Dataset']==2,'Dataset']=1
dataset['Dataset'].value_counts()

#For visualising
import seaborn as sns
#Counting the number of occurence of the different values of the Dataset column
sns.countplot(dataset['Dataset'])

#Checking the Average of the total proteins vs the Whether the person is liver patient or not.
sns.boxplot(x='Dataset',y='Total_Protiens',hue='Gender',data=dataset)

#Checking the Average of the Age vs the Whether the person is liver patient or not.
sns.boxplot(x='Dataset',y='Albumin',hue='Gender',data=dataset,width=0.6)

#Checking the Average of the Albumin vs the Whether the person is liver patient or not.
sns.boxplot(x='Dataset',y='Albumin',data=dataset,width=0.6)

#Converting the Categorical features
new_Data=pd.get_dummies(dataset,columns=['Gender'],drop_first=True)
new_Data.head()

#Checking for null data
sns.heatmap(new_Data.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#converting the null value into mean value
def Converter(data):
    if pd.isnull(data):
        return new_Data['Albumin_and_Globulin_Ratio'].mean()
    else:
        return data
#Applying the Function to column of the dataset that have null values
new_Data['Albumin_and_Globulin_Ratio']=new_Data['Albumin_and_Globulin_Ratio'].apply(Converter)
#Rechecking for null value
sns.heatmap(new_Data.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#Converting the dataset into x and y(target variable)
X=new_Data.drop('Dataset',axis=1)
y=new_Data['Dataset']
#Dividing the data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#feature scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
X_train
X_test

import keras
#importing the libraries for our ANN Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
#Initialising the model
model=Sequential()
#adding the first layer
model.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=10))
#new_Data.info()
#adding the second layer
model.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
#adding the output layer
model.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
#compiling all the layer together
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#Fitting the data to our model
model.fit(X_train,y_train,batch_size=64,epochs=400)
#Making Prediction from our model
predictions=model.predict(X_test)
#converting the probablitiy obtained using the predict method to the binary output
predictions=(predictions>0.5)
#Importing the library and Evaluating the performance of our ANN Model on the test set
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print('Confusion Matrix \n',confusion_matrix(y_test,predictions))

#Printing the Classifcation Report
print('Classification Report \n',classification_report(y_test,predictions))

#Printing the Classifcation Report
print('Accuracy of our mode when applied on test set-',accuracy_score(y_test,predictions))


Building Artificial Neural Network for handwritten digits classification

import tensorflow as tf
from tensorflow import keras
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
#load the mnist dataset
(x_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
len(x_train)

x_train[0].shape
x_train[0]
’
plt.matshow(x_train[225])

plt.matshow(x_train[0])
plt.matshow(x_train[4])

y_train[:5]

x_train.shape
x_train_flattened = x_train.reshape(len(x_train), 28*28)
x_train_flattened.shape
x_test_flattened = X_test.reshape(len(X_test), 28*28)
x_test_flattened.shape
#to improve accuracy lets scale the x_train and y_train
#scale is the techniques which improve accuracy of a model
x_train = x_train/255
X_test = X_test/255
model = keras.Sequential([
    keras.layers.Dense(10,input_shape=(784,), activation='sigmoid')


])
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(x_train_flattened, y_train, epochs=5)

model.evaluate(x_test_flattened, y_test)
model.predict(x_test_flattened)
y_predicted = model.predict(x_test_flattened)
y_predicted[0]
y_test[0]
tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)


plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


Implementation of Convolutional neural networks on Cifar dataset

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
(X_train,y_train),(X_test, y_test) = datasets.cifar10.load_data()
X_train.shape

X_test.shape

y_train.shape
y_train[:5]
y_train = y_train.reshape(-1,)
y_train[:5]
y_test = y_test.reshape(-1,)
classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
def plot_sample(X,y,index):
    plt.figure(figsize=(15,2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])
plot_sample(X_train,y_train,0)

plot_sample(X_train,y_train,1)

X_train = X_train / 255.0
X_test = X_test / 255.0
ann = models.Sequential([
        layers.Flatten(input_shape=(32,32,3)),
        layers.Dense(3000, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(10, activation='sigmoid')
])
ann.compile(optimizer='SGD',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])


ann.fit(X_train, y_train, epochs=5)

from sklearn.metrics import confusion_matrix , classification_report
import numpy as np
y_pred = ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]


print("Classification Report: \n", classification_report(y_test, y_pred_classes))

cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),


    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),


    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
cnn.fit(X_train, y_train, epochs=10)

cnn.evaluate(X_test,y_test)
y_pred = cnn.predict(X_test)
y_pred[:5]
y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]


y_test[:5]
plot_sample(X_test, y_test,3)
classes[y_classes[3]]

Predicting House Prices using ANN

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
# Loading the dataset
data = pd.read_csv('/content/train.csv')
data.head()
# Seperating target variables
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']
# Handle missing values
for column in X.columns:
    if X[column].dtype == 'float64' or X[column].dtype == 'int64':
        X[column].fillna(X[column].mean(), inplace=True)
    else:
        X[column].fillna(X[column].mode()[0], inplace=True)
# One-hot encoding
X = pd.get_dummies(X)
# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Build the ANN model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output layer for regression
])
# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])
# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f"Mean Absolute Error: {mae}")
# R-squared using sklearn
from sklearn.metrics import r2_score
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")


 Classification of Handwritten Digits using CNN
 
 import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train, _, y_train, _ = train_test_split(x_train, y_train, test_size=0.83, random_state=42)
x_test, _, y_test, _ = train_test_split(x_test, y_test, test_size=0.90, random_state=42)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))




model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")
plt.plot(history.history['loss'], label='Loss Training')
plt.plot(history.history['val_loss'], label='Loss Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


Image Classification of Animals using CNN

!pip install kaggle
! mkdir ~/.kaggle
!cp /content/kaggle.json ~/.kaggle/kaggle.json
!kaggle datasets download erkamk/cat-and-dog-images-dataset
! unzip /content/cat-and-dog-images-dataset.zip
import numpy as np
import pandas as pd
from pathlib import Path
import os.path


import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split


import tensorflow as tf


from sklearn.metrics import confusion_matrix, classification_report
image_dir = Path('/content/Dog and Cat .png')
train_df, test_df = train_test_split(image_df, train_size=0.7, shuffle=True, random_state=1)
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split=0.2
)


test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)
train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='binary',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='training'
)
val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='binary',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='validation'
)


test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='binary',
    batch_size=32,
    shuffle=False
)
inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(inputs)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)


model = tf.keras.Model(inputs=inputs, outputs=outputs)


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=100,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=3
        )
    ]
)



Predicting Customer Churn using ANN

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
data =pd.read_csv('/content/WA_Fn-UseC_-Telco-Customer-Churn.csv')
data.head()
data.describe()
data.info()
# Preprocess the data
# Convert categorical variables to numerical using LabelEncoder
le = LabelEncoder()
for column in data.columns:
    if data[column].dtype == object:
        data[column] = le.fit_transform(data[column])


# Separate features (X) and target variable (y)
X = data.drop('customerID', axis=1)
y = data['Churn']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Normalize numerical data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Build the ANN model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)
# Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype("int32")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

Chest X-ray Image Classification using CNN

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os
import random


# Set parameters
img_height, img_width = 150, 150
batch_size = 32
num_classes = 3


# Paths to training and validation datasets
train_data_dir = '/content/drive/MyDrive/XRAYS'  # Replace with your training data path
validation_data_dir = '/content/drive/MyDrive/XRAYS'  # Replace with your validation data path
# Prepare the data
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'  # Change to 'categorical' for multi-class
)


validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'  # Change to 'categorical' for multi-class
)
# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # Change to 'softmax' for multi-class
])


# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Change to 'categorical_crossentropy' for multi-class
              metrics=['accuracy'])
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=100  # Adjust number of epochs as needed
)




# Predict the classes for the validation set
Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)


# Get the true labels from the validation generator
y_true = validation_generator.classes


# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f'Overall Accuracy: {accuracy:.4f}')


# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=validation_generator.class_indices.keys()))


# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)


# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=validation_generator.class_indices.keys())
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
def plot_history(history):
    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs = range(len(acc))


    plt.figure(figsize=(12, 8))


    # Accuracy Plot
    plt.subplot(2, 1, 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()


    # Loss Plot
    plt.subplot(2, 1, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()


    plt.tight_layout()
    plt.show()


    # Check if validation metrics are available
    if 'val_accuracy' in history.history:
        val_acc = history.history['val_accuracy']
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
        plt.title('Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
    if 'val_loss' in history.history:
        val_loss = history.history['val_loss']
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


plot_history(history)
from tensorflow.keras.preprocessing import image


def predict_random_image(model, image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale the image


    # Make predictions
    prediction = model.predict(img_array)


    # Get the predicted class index
    predicted_class_index = np.argmax(prediction, axis=1)[0]


    # Class names
    class_names = ['Normal', 'Viral Pneumonia', 'COVID']  # Adjusted for your classes


    # Determine predicted class
    predicted_class = class_names[predicted_class_index]


    # Show the image and prediction
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Predicted: {predicted_class}')
    plt.show()


# Example usage: predict a specific image
random_image_path = "/content/drive/MyDrive/XRAYS/Viral Pneumonia/0101.jpeg"  # Change to your image path
predict_random_image(model, random_image_path)
# Example usage: predict a specific image
random_image_path = "/content/drive/MyDrive/XRAYS/Normal/0112.jpeg"  # Change to your image path
predict_random_image(model, random_image_path)


 Implementation of Basic Recurrent Neural Network
 
 # Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import SimpleRNN, Dense


# Generate a sample sequence
sequence = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


# Prepare the data
X = []
y = []


# Create input-output pairs from the sequence
for i in range(len(sequence) - 3):
    X.append(sequence[i:i+3])
    y.append(sequence[i+3])


X = np.array(X)
y = np.array(y)


# Define and compile the RNN model
model = keras.Sequential()
model.add(SimpleRNN(2, input_shape=(3, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')


# Train the model
model.fit(X, y, epochs=1000, verbose=0)


# Generate a new sequence based on the learned model
initial_sequence = [0.1, 0.2, 0.3]  # Initial part of the sequence
predicted_sequence = []


for _ in range(4):
    next_value = model.predict(np.array(initial_sequence[-3:]).reshape(1, 3, 1))
    predicted_sequence.append(next_value[0, 0])
    initial_sequence.append(next_value[0, 0])


# Print the predicted sequence
print("Initial Sequence:", initial_sequence[:3])
print("Predicted Sequence:", predicted_sequence)



