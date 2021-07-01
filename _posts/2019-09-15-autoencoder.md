---
layout: post
title: Anomaly detection with autoencoders (10 mins read) 
---


<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>


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
import plotly.offline as py_offline
```

    2021-07-01 14:47:04.153308: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
    2021-07-01 14:47:04.153324: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.



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

    2021-07-01 14:47:06.736378: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcuda.so.1
    2021-07-01 14:47:06.760692: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2021-07-01 14:47:06.760896: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1733] Found device 0 with properties: 
    pciBusID: 0000:01:00.0 name: Quadro T2000 computeCapability: 7.5
    coreClock: 1.5GHz coreCount: 16 deviceMemorySize: 3.82GiB deviceMemoryBandwidth: 104.34GiB/s
    2021-07-01 14:47:06.760960: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
    2021-07-01 14:47:06.761007: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcublas.so.11'; dlerror: libcublas.so.11: cannot open shared object file: No such file or directory
    2021-07-01 14:47:06.761046: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcublasLt.so.11'; dlerror: libcublasLt.so.11: cannot open shared object file: No such file or directory
    2021-07-01 14:47:06.761084: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcufft.so.10'; dlerror: libcufft.so.10: cannot open shared object file: No such file or directory
    2021-07-01 14:47:06.761121: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcurand.so.10'; dlerror: libcurand.so.10: cannot open shared object file: No such file or directory
    2021-07-01 14:47:06.761159: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcusolver.so.11'; dlerror: libcusolver.so.11: cannot open shared object file: No such file or directory
    2021-07-01 14:47:06.761195: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcusparse.so.11'; dlerror: libcusparse.so.11: cannot open shared object file: No such file or directory
    2021-07-01 14:47:06.761234: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudnn.so.8'; dlerror: libcudnn.so.8: cannot open shared object file: No such file or directory
    2021-07-01 14:47:06.761244: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1766] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
    Skipping registering GPU devices...
    2021-07-01 14:47:06.762164: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2021-07-01 14:47:06.762529: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1258] Device interconnect StreamExecutor with strength 1 edge matrix:
    2021-07-01 14:47:06.762538: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1264]      



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




```python
fig = go.Figure()
trace=go.Scatter(x=np.arange(140), y=normal_train_data[0],name='Normal')
trace2=go.Scatter(x=np.arange(140), y=normal_train_data[0],name='Normal')
data = [trace,trace2]
py_offline.plot(data, filename='basic-line', include_plotlyjs=False, output_type='div')

```
<div>                            <div id="ff64409b-bce8-4f84-bf69-e418a91727bf" class="plotly-graph-div" style="height:100%; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("ff64409b-bce8-4f84-bf69-e418a91727bf")) {                    Plotly.newPlot(                        "ff64409b-bce8-4f84-bf69-e418a91727bf",                        [{"name":"Normal","type":"scatter","x":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139],"y":[0.5703046321868896,0.4656165838241577,0.2905811667442322,0.17791584134101868,0.09538919478654861,0.08467857539653778,0.2019510418176651,0.3163002133369446,0.33732032775878906,0.41424882411956787,0.47070595622062683,0.4691905081272125,0.4776775538921356,0.48004090785980225,0.4702724516391754,0.4729926884174347,0.479171484708786,0.48027467727661133,0.46928870677948,0.46106863021850586,0.46148037910461426,0.4465829133987427,0.45325326919555664,0.4561009407043457,0.4489617943763733,0.44325318932533264,0.4340217411518097,0.4430723488330841,0.4325052499771118,0.43014901876449585,0.41454657912254333,0.4120652973651886,0.40420278906822205,0.4105454385280609,0.4082913100719452,0.40342992544174194,0.39648476243019104,0.3933204412460327,0.39158815145492554,0.3960387706756592,0.4074695408344269,0.40531179308891296,0.41734880208969116,0.41624557971954346,0.42334118485450745,0.4459063708782196,0.4455184042453766,0.44474098086357117,0.44404327869415283,0.4567321836948395,0.4521746337413788,0.4532504081726074,0.46082931756973267,0.4669593572616577,0.4663194417953491,0.47432830929756165,0.46297091245651245,0.4738894999027252,0.4678889811038971,0.4673447906970978,0.4802667498588562,0.48439839482307434,0.48574668169021606,0.48985567688941956,0.4931734800338745,0.48830410838127136,0.49913936853408813,0.505709171295166,0.5078241229057312,0.5132302641868591,0.5211992859840393,0.514991819858551,0.519900918006897,0.5074853897094727,0.5113434195518494,0.5092939734458923,0.5073481798171997,0.5110538005828857,0.5096123814582825,0.4963955879211426,0.49456536769866943,0.5014910101890564,0.5063670873641968,0.5020293593406677,0.5007152557373047,0.4958237409591675,0.4843623638153076,0.4914246201515198,0.4825357496738434,0.47723883390426636,0.47550636529922485,0.47530031204223633,0.4876338243484497,0.4776129722595215,0.474583238363266,0.4757544696331024,0.47009119391441345,0.4540967047214508,0.46874862909317017,0.4767007529735565,0.4795878529548645,0.4751480519771576,0.47619494795799255,0.48561891913414,0.48763763904571533,0.4996277689933777,0.5284430384635925,0.5431970953941345,0.5495933294296265,0.5488370060920715,0.5231477618217468,0.4934438467025757,0.49881476163864136,0.5099680423736572,0.516133189201355,0.4907889664173126,0.4668024778366089,0.43299439549446106,0.4164400100708008,0.4182245135307312,0.43140411376953125,0.43214115500450134,0.4212411642074585,0.42373567819595337,0.42885276675224304,0.43089887499809265,0.43537637591362,0.4391244351863861,0.4371418356895447,0.4453428089618683,0.4533092975616455,0.48821336030960083,0.5786804556846619,0.5858615636825562,0.5959517955780029,0.5952476263046265,0.5700759291648865,0.4850423336029053,0.42335018515586853,0.4759834408760071]},{"name":"Normal","type":"scatter","x":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139],"y":[0.5703046321868896,0.4656165838241577,0.2905811667442322,0.17791584134101868,0.09538919478654861,0.08467857539653778,0.2019510418176651,0.3163002133369446,0.33732032775878906,0.41424882411956787,0.47070595622062683,0.4691905081272125,0.4776775538921356,0.48004090785980225,0.4702724516391754,0.4729926884174347,0.479171484708786,0.48027467727661133,0.46928870677948,0.46106863021850586,0.46148037910461426,0.4465829133987427,0.45325326919555664,0.4561009407043457,0.4489617943763733,0.44325318932533264,0.4340217411518097,0.4430723488330841,0.4325052499771118,0.43014901876449585,0.41454657912254333,0.4120652973651886,0.40420278906822205,0.4105454385280609,0.4082913100719452,0.40342992544174194,0.39648476243019104,0.3933204412460327,0.39158815145492554,0.3960387706756592,0.4074695408344269,0.40531179308891296,0.41734880208969116,0.41624557971954346,0.42334118485450745,0.4459063708782196,0.4455184042453766,0.44474098086357117,0.44404327869415283,0.4567321836948395,0.4521746337413788,0.4532504081726074,0.46082931756973267,0.4669593572616577,0.4663194417953491,0.47432830929756165,0.46297091245651245,0.4738894999027252,0.4678889811038971,0.4673447906970978,0.4802667498588562,0.48439839482307434,0.48574668169021606,0.48985567688941956,0.4931734800338745,0.48830410838127136,0.49913936853408813,0.505709171295166,0.5078241229057312,0.5132302641868591,0.5211992859840393,0.514991819858551,0.519900918006897,0.5074853897094727,0.5113434195518494,0.5092939734458923,0.5073481798171997,0.5110538005828857,0.5096123814582825,0.4963955879211426,0.49456536769866943,0.5014910101890564,0.5063670873641968,0.5020293593406677,0.5007152557373047,0.4958237409591675,0.4843623638153076,0.4914246201515198,0.4825357496738434,0.47723883390426636,0.47550636529922485,0.47530031204223633,0.4876338243484497,0.4776129722595215,0.474583238363266,0.4757544696331024,0.47009119391441345,0.4540967047214508,0.46874862909317017,0.4767007529735565,0.4795878529548645,0.4751480519771576,0.47619494795799255,0.48561891913414,0.48763763904571533,0.4996277689933777,0.5284430384635925,0.5431970953941345,0.5495933294296265,0.5488370060920715,0.5231477618217468,0.4934438467025757,0.49881476163864136,0.5099680423736572,0.516133189201355,0.4907889664173126,0.4668024778366089,0.43299439549446106,0.4164400100708008,0.4182245135307312,0.43140411376953125,0.43214115500450134,0.4212411642074585,0.42373567819595337,0.42885276675224304,0.43089887499809265,0.43537637591362,0.4391244351863861,0.4371418356895447,0.4453428089618683,0.4533092975616455,0.48821336030960083,0.5786804556846619,0.5858615636825562,0.5959517955780029,0.5952476263046265,0.5700759291648865,0.4850423336029053,0.42335018515586853,0.4759834408760071]}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}}},                        {"responsive": true}                    )                };                            </script>        </div>



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

    2021-07-01 14:47:07.260279: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:176] None of the MLIR Optimization Passes are enabled (registered 2)
    2021-07-01 14:47:07.261000: I tensorflow/core/platform/profile_utils/cpu_utils.cc:114] CPU Frequency: 2299965000 Hz
    Epoch 1/40
    3/3 [==============================] - 0s 49ms/step - loss: 0.0595 - val_loss: 0.0544
    Epoch 2/40
    3/3 [==============================] - 0s 10ms/step - loss: 0.0577 - val_loss: 0.0536
    Epoch 3/40
    3/3 [==============================] - 0s 12ms/step - loss: 0.0567 - val_loss: 0.0528
    Epoch 4/40
   

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



Let's do the same as above for an abnormal rhythm ECG on the test dataset.


```python
encoded_data = autoencoder.encoder(abnormal_test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()


fig = go.Figure()
fig.add_trace(go.Scatter(y=abnormal_test_data[0],name='Input data'))
fig.add_trace(go.Scatter(y=decoded_data[0],name='Reconstruction & error',
                         fill='tonexty'))

```



With the naked eye, the two plots above seem to suggest that the reconstruction error for the abnormal rhythm ECG is larger. We will formalize our findings in the next section.

## Detecting anomalies

Here we will compute the reconstruction error for all the data points both normal and abnormal.  For the reconstruction error, we will use the mean absolute error.
We will compute the reconstruction error of the training dataset and choose a threshold  (one standard deviation away from the mean) above which we will classify an ECG as abnormal.



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

    Threshold:  0.032012627


On the test dataset, we will use the threshold above to determine anormalies. We will do this as follows:


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

In this blog post, we have seen how autoencoders can be used to detect anomalies in our data. The ECG data is a  nice example to illustrate the idea, however, with a typical real-world use case, there will be more shortcomings. 
