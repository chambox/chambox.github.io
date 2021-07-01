## Anomaly detection with autoencoders

In this blog post we will use autoencoders to detect anomalies in ECG5000 dataset. In writing this blog post I have assumed that the reader has some basic understanding of neural networks and autoencoders as a specific type of neural network. Jeremy Jordan has a nice 10mins-read blog post on autoencoders, you can find it [here](https://www.jeremyjordan.me/autoencoders/).  The ECG5000 dataset which we will use contains 50,000 Electrocardiograms (ECG). Each cardiogram has 140 data points.  Luckily for us, the data has been labeled into a normal and abnormal rhythm by medical experts. Our goal is to use autoencoders to see if they can mimic the knowledge of a medical doctor and identify abnormal Electrocardiograms.

Our approach will be to (1) train the autoencoder on the normal data and (2) use our trained model to reconstruct the entire dataset. We hypothesize that abnormal Electrocardiograms will have a higher reconstruction error. Recall that an autoencoder takes the input data and projects it onto a lower-dimensional space that captures only the signals in the data. The data can then be reconstructed from the lower-dimensional space. Note here that if a data point is noisy its reconstruction error (the distance between the actual point and the reconstructed one) will be large. It is this simple principle that we use to identity anomalies. 


## Lets load the ECG data 


```python
# libraries 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
import plotly.express as px 
import plotly.graph_objects as go 
```


```python
# Download the dataset
url = 'http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv'

dataframe = pd.read_csv(url,header=None)
raw_data = dataframe.values
dataframe.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>131</th>
      <th>132</th>
      <th>133</th>
      <th>134</th>
      <th>135</th>
      <th>136</th>
      <th>137</th>
      <th>138</th>
      <th>139</th>
      <th>140</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.112522</td>
      <td>-2.827204</td>
      <td>-3.773897</td>
      <td>-4.349751</td>
      <td>-4.376041</td>
      <td>-3.474986</td>
      <td>-2.181408</td>
      <td>-1.818286</td>
      <td>-1.250522</td>
      <td>-0.477492</td>
      <td>...</td>
      <td>0.792168</td>
      <td>0.933541</td>
      <td>0.796958</td>
      <td>0.578621</td>
      <td>0.257740</td>
      <td>0.228077</td>
      <td>0.123431</td>
      <td>0.925286</td>
      <td>0.193137</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.100878</td>
      <td>-3.996840</td>
      <td>-4.285843</td>
      <td>-4.506579</td>
      <td>-4.022377</td>
      <td>-3.234368</td>
      <td>-1.566126</td>
      <td>-0.992258</td>
      <td>-0.754680</td>
      <td>0.042321</td>
      <td>...</td>
      <td>0.538356</td>
      <td>0.656881</td>
      <td>0.787490</td>
      <td>0.724046</td>
      <td>0.555784</td>
      <td>0.476333</td>
      <td>0.773820</td>
      <td>1.119621</td>
      <td>-1.436250</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.567088</td>
      <td>-2.593450</td>
      <td>-3.874230</td>
      <td>-4.584095</td>
      <td>-4.187449</td>
      <td>-3.151462</td>
      <td>-1.742940</td>
      <td>-1.490659</td>
      <td>-1.183580</td>
      <td>-0.394229</td>
      <td>...</td>
      <td>0.886073</td>
      <td>0.531452</td>
      <td>0.311377</td>
      <td>-0.021919</td>
      <td>-0.713683</td>
      <td>-0.532197</td>
      <td>0.321097</td>
      <td>0.904227</td>
      <td>-0.421797</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.490473</td>
      <td>-1.914407</td>
      <td>-3.616364</td>
      <td>-4.318823</td>
      <td>-4.268016</td>
      <td>-3.881110</td>
      <td>-2.993280</td>
      <td>-1.671131</td>
      <td>-1.333884</td>
      <td>-0.965629</td>
      <td>...</td>
      <td>0.350816</td>
      <td>0.499111</td>
      <td>0.600345</td>
      <td>0.842069</td>
      <td>0.952074</td>
      <td>0.990133</td>
      <td>1.086798</td>
      <td>1.403011</td>
      <td>-0.383564</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.800232</td>
      <td>-0.874252</td>
      <td>-2.384761</td>
      <td>-3.973292</td>
      <td>-4.338224</td>
      <td>-3.802422</td>
      <td>-2.534510</td>
      <td>-1.783423</td>
      <td>-1.594450</td>
      <td>-0.753199</td>
      <td>...</td>
      <td>1.148884</td>
      <td>0.958434</td>
      <td>1.059025</td>
      <td>1.371682</td>
      <td>1.277392</td>
      <td>0.960304</td>
      <td>0.971020</td>
      <td>1.614392</td>
      <td>1.421456</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 141 columns</p>
</div>



The last column contains the labels. The other data points are the electrocardiogram data. We will also create a train and test a dataset, as well as their labels, this is what data scientists do to overcome a serious problem in data science namely `over-fitting`.


```python
# The last element contains the labels
labels = raw_data[:, -1]

# The other data points are the electrocadriogram data
data = raw_data[:, 0:-1]

train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=21
)
```

We will also normalize the data to lie between [0,1]. Note here that we do normalization because to speed up the learning and convergence of the optimizer. 



```python
min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)

train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)
```


As mentioned earlier, we will only train the autoencoder on the data with normal rhythms. Electrocardiograms with normal rhythm are labeled with 1. We will separate the normal rhythm from the abnormal ones in the following chunk of code.


```python
train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]

abnormal_train_data = train_data[~train_labels]
abnormal_test_data = test_data[~test_labels]
```

Let's visualize a normal and an abnormal  ECG. 


```python
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(140), y=normal_train_data[0],name='Normal'))
fig.add_trace(go.Scatter(x=np.arange(140), y=abnormal_train_data[10],name='Abnormal'))

```



Observe that there is a huge discrepancy between the normal and abnormal graphs for large values on the x-axis. We will now build an autoencoder that will encode the normal data.

Build the model. Here we will use `Kera`'s sequential API, with three dense layers for the encoder and three dense layers for the decoder. We will create an `AnormlyDectector` class that inherits from the `Model` class. 


```python
class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(32, activation="relu"),
      layers.Dense(16, activation="relu"),
      layers.Dense(8, activation="relu")])

    self.decoder = tf.keras.Sequential([
      layers.Dense(16, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(140, activation="sigmoid")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = AnomalyDetector()
```

Note that the call method in the `AnormalyDetector` class combines the `encoder` and `decoder` and returns the `decoder` object. Let's `compile`, compilation here will mean we will update our `autoencoder` object with an `optimizer` and a `loss` function. We are using the mean absolute error loss defined as:

$$
\text{mae} = \frac{1}{n}\sum_{i=1}^n{|y_i-\hat{y}_i|}.
$$

Where in this simple formular, we have $n$ data points $i = 1,2,...,n$, $y_i$ refers to the actual (true/observed) data point and $\hat{y}_i$ is its estimate. In our use case, $y_i$ is the actual ECG and $\hat{y}_i$ will be its reconstructed version.


```python
autoencoder.compile(optimizer='adam', loss='mae')
```

We now train the `autoencoder` by calling its fit method using only the normal rhythm ECG.


```python
history = autoencoder.fit(normal_train_data, normal_train_data, 
          epochs=40, 
          batch_size=1024,
          validation_data=(test_data, test_data),
          shuffle=True)
```

    Epoch 1/40
    3/3 [==============================] - 1s 57ms/step - loss: 0.0582 - val_loss: 0.0536
    Epoch 2/40
    3/3 [==============================] - 0s 13ms/step - loss: 0.0564 - val_loss: 0.0523
    Epoch 3/40
    3/3 [==============================] - 0s 12ms/step - loss: 0.0548 - val_loss: 0.0512
    Epoch 4/40
    3/3 [==============================] - 0s 12ms/step - loss: 0.0529 - val_loss: 0.0503
    Epoch 5/40
    3/3 [==============================] - 0s 11ms/step - loss: 0.0510 - val_loss: 0.0493
    Epoch 6/40
    3/3 [==============================] - 0s 38ms/step - loss: 0.0490 - val_loss: 0.0480
    Epoch 7/40
    3/3 [==============================] - 0s 35ms/step - loss: 0.0469 - val_loss: 0.0468
    Epoch 8/40
    3/3 [==============================] - 0s 37ms/step - loss: 0.0448 - val_loss: 0.0459
    Epoch 9/40
    3/3 [==============================] - 0s 38ms/step - loss: 0.0426 - val_loss: 0.0449
    Epoch 10/40
    3/3 [==============================] - 0s 43ms/step - loss: 0.0406 - val_loss: 0.0438
    Epoch 11/40
    3/3 [==============================] - 0s 35ms/step - loss: 0.0386 - val_loss: 0.0429
    Epoch 12/40
    3/3 [==============================] - 0s 38ms/step - loss: 0.0368 - val_loss: 0.0422
    Epoch 13/40
    3/3 [==============================] - 0s 27ms/step - loss: 0.0352 - val_loss: 0.0414
    Epoch 14/40
    3/3 [==============================] - 0s 30ms/step - loss: 0.0337 - val_loss: 0.0407
    Epoch 15/40
    3/3 [==============================] - 0s 30ms/step - loss: 0.0322 - val_loss: 0.0402
    Epoch 16/40
    3/3 [==============================] - 0s 31ms/step - loss: 0.0309 - val_loss: 0.0394
    Epoch 17/40
    3/3 [==============================] - 0s 37ms/step - loss: 0.0298 - val_loss: 0.0390
    Epoch 18/40
    3/3 [==============================] - 0s 36ms/step - loss: 0.0289 - val_loss: 0.0385
    Epoch 19/40
    3/3 [==============================] - 0s 33ms/step - loss: 0.0281 - val_loss: 0.0380
    Epoch 20/40
    3/3 [==============================] - 0s 15ms/step - loss: 0.0274 - val_loss: 0.0379
    Epoch 21/40
    3/3 [==============================] - 0s 15ms/step - loss: 0.0269 - val_loss: 0.0373
    Epoch 22/40
    3/3 [==============================] - 0s 15ms/step - loss: 0.0263 - val_loss: 0.0370
    Epoch 23/40
    3/3 [==============================] - 0s 14ms/step - loss: 0.0258 - val_loss: 0.0366
    Epoch 24/40
    3/3 [==============================] - 0s 14ms/step - loss: 0.0254 - val_loss: 0.0362
    Epoch 25/40
    3/3 [==============================] - 0s 15ms/step - loss: 0.0249 - val_loss: 0.0357
    Epoch 26/40
    3/3 [==============================] - 0s 15ms/step - loss: 0.0245 - val_loss: 0.0354
    Epoch 27/40
    3/3 [==============================] - 0s 14ms/step - loss: 0.0241 - val_loss: 0.0352
    Epoch 28/40
    3/3 [==============================] - 0s 15ms/step - loss: 0.0237 - val_loss: 0.0346
    Epoch 29/40
    3/3 [==============================] - 0s 15ms/step - loss: 0.0234 - val_loss: 0.0343
    Epoch 30/40
    3/3 [==============================] - 0s 14ms/step - loss: 0.0230 - val_loss: 0.0341
    Epoch 31/40
    3/3 [==============================] - 0s 14ms/step - loss: 0.0227 - val_loss: 0.0337
    Epoch 32/40
    3/3 [==============================] - 0s 14ms/step - loss: 0.0223 - val_loss: 0.0334
    Epoch 33/40
    3/3 [==============================] - 0s 16ms/step - loss: 0.0220 - val_loss: 0.0330
    Epoch 34/40
    3/3 [==============================] - 0s 14ms/step - loss: 0.0217 - val_loss: 0.0329
    Epoch 35/40
    3/3 [==============================] - 0s 14ms/step - loss: 0.0214 - val_loss: 0.0327
    Epoch 36/40
    3/3 [==============================] - 0s 14ms/step - loss: 0.0212 - val_loss: 0.0324
    Epoch 37/40
    3/3 [==============================] - 0s 14ms/step - loss: 0.0209 - val_loss: 0.0324
    Epoch 38/40
    3/3 [==============================] - 0s 12ms/step - loss: 0.0207 - val_loss: 0.0320
    Epoch 39/40
    3/3 [==============================] - 0s 12ms/step - loss: 0.0205 - val_loss: 0.0318
    Epoch 40/40
    3/3 [==============================] - 0s 12ms/step - loss: 0.0204 - val_loss: 0.0318


Note that although, the training is done on the normal rythm ECG, the validation is done on the entire test dataset. 


```python
fig = go.Figure()
fig.add_trace(go.Scatter(y=history.history["loss"],name='Training loss'))
fig.add_trace(go.Scatter(y=history.history["val_loss"],name="Validation Loss"))
```



## Reconstruction error

We now have a model that can encode and decode ECG. Let's use the model to reconstruct a particular ECG and check the reconstruction error, i.e., the difference between the actual ECG and its reconstruction. 



```python
encoded_data = autoencoder.encoder(normal_test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()


fig = go.Figure()
fig.add_trace(go.Scatter(y=normal_test_data[0],name='Input data'))
fig.add_trace(go.Scatter(y=decoded_data[0],name='Reconstruction & error',
                         fill='tonexty'))

```



Let's do the same as above for an abnormal rythm ECG on the test dataset.


```python
encoded_data = autoencoder.encoder(abnormal_test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()


fig = go.Figure()
fig.add_trace(go.Scatter(y=abnormal_test_data[0],name='Input data'))
fig.add_trace(go.Scatter(y=decoded_data[0],name='Reconstruction & error',
                         fill='tonexty'))

```



With the naked eye, the two plots above seem to suggest that the reconstruction error for the abnormal rythm ECG is larger. We will formalise our findings in the next section.

## Detecting anomalies

Here we will compute the reconstruction error for all the data points both normal and abnormal.  For the reconstruction error we will use the mean absolute error.
We will compute the reconstruction error of the training dataset and choose a threshold  (one standard deviation away from the mean) above which we will class ECG as abnormal.



```python
reconstructions = autoencoder.predict(normal_train_data)
train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)

fig = go.Figure()
fig.add_trace(go.Histogram(x=train_loss[None,:][0],name='Normal loss'))
```



We now define the threshold.


```python
threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold)
```

    Threshold:  0.0318527


On the test dataset we will use the threshold above to determine abnormalies. We will do this as follows:


```python
reconstructions_normal = autoencoder.predict(normal_test_data)
test_loss_normal = tf.keras.losses.mae(reconstructions_normal, normal_test_data)


reconstructions_abnormal = autoencoder.predict(abnormal_test_data)
test_loss_abnormal = tf.keras.losses.mae(reconstructions_abnormal, abnormal_test_data)

fig = go.Figure()
fig.add_trace(go.Histogram(x=test_loss_normal[None,:][0],name='Normal loss'))
fig.add_trace(go.Histogram(x=test_loss_abnormal[None,:][0],name='Abnormal loss'))


# Overlay both histograms
fig.update_layout(barmode='overlay')
# Reduce opacity to see both histograms
fig.update_traces(opacity=0.75)




fig.add_shape(type="line",
    xref="x", yref="y",
    x0=threshold, y0=0, x1=threshold, y1=90,
    line=dict(
        color="red",
        width=3,
    ),
)
fig.show()
```



The red vertical line is at the threshold. Anything above the red vertical line is considered as an anormaly. 

## How accurate is our model 

We will compute the accuracy, precision and recall of our model. 


```python
def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mae(reconstructions, data)
  return tf.math.less(loss, threshold)

def print_stats(predictions, labels):
  print("Accuracy = {}".format(accuracy_score(labels, predictions)))
  print("Precision = {}".format(precision_score(labels, predictions)))
  print("Recall = {}".format(recall_score(labels, predictions)))



preds = predict(autoencoder, test_data, threshold)
print_stats(preds, test_labels)
```

    Accuracy = 0.943
    Precision = 0.9921722113502935
    Recall = 0.9053571428571429


## Final words

In this blog post we have seen how autoencoders can be used to detect anomaly in our data. The ECG data is a very nice example to illustrate the idea. 


```python

```
