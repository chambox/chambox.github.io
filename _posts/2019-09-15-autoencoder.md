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


```python
# Download the dataset
url = 'http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv'

dataframe = pd.read_csv(url,header=None)
raw_data = dataframe.values
# first five rows and columns
print(dataframe.iloc[:,0:5].head().to_markdown())
```

    |    |         0 |         1 |        2 |        3 |        4 |
    |---:|----------:|----------:|---------:|---------:|---------:|
    |  0 | -0.112522 | -2.8272   | -3.7739  | -4.34975 | -4.37604 |
    |  1 | -1.10088  | -3.99684  | -4.28584 | -4.50658 | -4.02238 |
    |  2 | -0.567088 | -2.59345  | -3.87423 | -4.58409 | -4.18745 |
    |  3 |  0.490473 | -1.91441  | -3.61636 | -4.31882 | -4.26802 |
    |  4 |  0.800232 | -0.874252 | -2.38476 | -3.97329 | -4.33822 |


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
trace1 = go.Scatter(x=np.arange(140), y=normal_train_data[0],name='Normal')
trace2 = go.Scatter(x=np.arange(140), y=abnormal_train_data[0],name='Abnormal')
data = [trace1,trace2]
py_offline.plot(data, filename='basic-line', include_plotlyjs=False, output_type='div')


```
 <div>                            <div id="df570e40-a9ee-4b30-9e74-8952a20a474b" class="plotly-graph-div" style="height:100%; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("df570e40-a9ee-4b30-9e74-8952a20a474b")) {                    Plotly.newPlot(                        "df570e40-a9ee-4b30-9e74-8952a20a474b",                        [{"name":"Normal","type":"scatter","x":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139],"y":[0.5703046321868896,0.4656165838241577,0.2905811667442322,0.17791584134101868,0.09538919478654861,0.08467857539653778,0.2019510418176651,0.3163002133369446,0.33732032775878906,0.41424882411956787,0.47070595622062683,0.4691905081272125,0.4776775538921356,0.48004090785980225,0.4702724516391754,0.4729926884174347,0.479171484708786,0.48027467727661133,0.46928870677948,0.46106863021850586,0.46148037910461426,0.4465829133987427,0.45325326919555664,0.4561009407043457,0.4489617943763733,0.44325318932533264,0.4340217411518097,0.4430723488330841,0.4325052499771118,0.43014901876449585,0.41454657912254333,0.4120652973651886,0.40420278906822205,0.4105454385280609,0.4082913100719452,0.40342992544174194,0.39648476243019104,0.3933204412460327,0.39158815145492554,0.3960387706756592,0.4074695408344269,0.40531179308891296,0.41734880208969116,0.41624557971954346,0.42334118485450745,0.4459063708782196,0.4455184042453766,0.44474098086357117,0.44404327869415283,0.4567321836948395,0.4521746337413788,0.4532504081726074,0.46082931756973267,0.4669593572616577,0.4663194417953491,0.47432830929756165,0.46297091245651245,0.4738894999027252,0.4678889811038971,0.4673447906970978,0.4802667498588562,0.48439839482307434,0.48574668169021606,0.48985567688941956,0.4931734800338745,0.48830410838127136,0.49913936853408813,0.505709171295166,0.5078241229057312,0.5132302641868591,0.5211992859840393,0.514991819858551,0.519900918006897,0.5074853897094727,0.5113434195518494,0.5092939734458923,0.5073481798171997,0.5110538005828857,0.5096123814582825,0.4963955879211426,0.49456536769866943,0.5014910101890564,0.5063670873641968,0.5020293593406677,0.5007152557373047,0.4958237409591675,0.4843623638153076,0.4914246201515198,0.4825357496738434,0.47723883390426636,0.47550636529922485,0.47530031204223633,0.4876338243484497,0.4776129722595215,0.474583238363266,0.4757544696331024,0.47009119391441345,0.4540967047214508,0.46874862909317017,0.4767007529735565,0.4795878529548645,0.4751480519771576,0.47619494795799255,0.48561891913414,0.48763763904571533,0.4996277689933777,0.5284430384635925,0.5431970953941345,0.5495933294296265,0.5488370060920715,0.5231477618217468,0.4934438467025757,0.49881476163864136,0.5099680423736572,0.516133189201355,0.4907889664173126,0.4668024778366089,0.43299439549446106,0.4164400100708008,0.4182245135307312,0.43140411376953125,0.43214115500450134,0.4212411642074585,0.42373567819595337,0.42885276675224304,0.43089887499809265,0.43537637591362,0.4391244351863861,0.4371418356895447,0.4453428089618683,0.4533092975616455,0.48821336030960083,0.5786804556846619,0.5858615636825562,0.5959517955780029,0.5952476263046265,0.5700759291648865,0.4850423336029053,0.42335018515586853,0.4759834408760071]},{"name":"Abnormal","type":"scatter","x":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139],"y":[0.4304001033306122,0.35345321893692017,0.3034263849258423,0.2818489074707031,0.28353944420814514,0.28959953784942627,0.3114522695541382,0.3470645248889923,0.3809654414653778,0.39506226778030396,0.3976452350616455,0.41356122493743896,0.4387162923812866,0.4520491063594818,0.4495706260204315,0.44956129789352417,0.45499545335769653,0.45426806807518005,0.44863343238830566,0.45084789395332336,0.45330581068992615,0.4520796537399292,0.4475529193878174,0.44939249753952026,0.4473649561405182,0.44874852895736694,0.4485017955303192,0.4462437331676483,0.44809791445732117,0.4483480751514435,0.44390979409217834,0.44208911061286926,0.4433857798576355,0.44462850689888,0.43986889719963074,0.4404641091823578,0.4400186240673065,0.43668332695961,0.4339315891265869,0.4317610263824463,0.4295136630535126,0.4274426996707916,0.42877644300460815,0.42894187569618225,0.4254743754863739,0.42517906427383423,0.42631953954696655,0.4277213215827942,0.42250746488571167,0.42367860674858093,0.4281102418899536,0.4270593523979187,0.4272007346153259,0.4308776557445526,0.4309622049331665,0.43490204215049744,0.43578556180000305,0.43896228075027466,0.44403478503227234,0.4491182863712311,0.453519344329834,0.4524179995059967,0.45339763164520264,0.45347079634666443,0.45640361309051514,0.45917513966560364,0.45685112476348877,0.45898863673210144,0.4569636881351471,0.45675793290138245,0.46130383014678955,0.45900118350982666,0.46151021122932434,0.46561598777770996,0.4587060809135437,0.4582265317440033,0.4626377522945404,0.4621920883655548,0.4627910554409027,0.46581798791885376,0.4644567370414734,0.4647052586078644,0.46456974744796753,0.4647565186023712,0.46593987941741943,0.46568334102630615,0.46936455368995667,0.47205060720443726,0.4691234230995178,0.4770551025867462,0.4743116497993469,0.4712502956390381,0.476502001285553,0.4755341112613678,0.47373101115226746,0.47748467326164246,0.48040735721588135,0.47647133469581604,0.4788639545440674,0.48452094197273254,0.4846736788749695,0.4795249104499817,0.4818657338619232,0.4819636642932892,0.48308029770851135,0.47801387310028076,0.48302972316741943,0.48857638239860535,0.48736441135406494,0.49351075291633606,0.49945521354675293,0.505765438079834,0.5169975757598877,0.527177631855011,0.5347791314125061,0.544451892375946,0.5556207895278931,0.5749051570892334,0.5851321816444397,0.5897799730300903,0.5978419184684753,0.6050950884819031,0.6238124370574951,0.6344885230064392,0.6365599036216736,0.6482247710227966,0.6608508229255676,0.6543368697166443,0.6259334683418274,0.5985985398292542,0.5772880911827087,0.5069698095321655,0.423368364572525,0.395474374294281,0.3666984438896179,0.35847392678260803,0.34862595796585083,0.3050689399242401,0.2622550129890442,0.25020110607147217]}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}}},                        {"responsive": true}                    )                };                            </script>        </div>



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

Where in this simple formular, we have $$ n $$ data points $$ i = 1,2,...,n$$, $$y_i $$ refers to the actual (true/observed) data point and $$\hat{y}_i$$ is its estimate. In our use case, $$y_i$$ is the actual ECG and $$\hat{y}_i$$ will be its reconstructed version.


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

    2021-07-01 23:30:53.389957: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:176] None of the MLIR Optimization Passes are enabled (registered 2)
    2021-07-01 23:30:53.407961: I tensorflow/core/platform/profile_utils/cpu_utils.cc:114] CPU Frequency: 2299965000 Hz
    Epoch 1/40
    3/3 [==============================] - 1s 68ms/step - loss: 0.0583 - val_loss: 0.0538
    Epoch 2/40
    3/3 [==============================] - 0s 12ms/step - loss: 0.0572 - val_loss: 0.0529
    Epoch 3/40
    3/3 [==============================] - 0s 13ms/step - loss: 0.0560 - val_loss: 0.0516
    Epoch 4/40
   


Note that although, the training is done on the normal rythm ECG, the validation is done on the entire test dataset. 


```python
trace1 = go.Scatter(y=history.history["loss"],name='Training loss')
trace2 = go.Scatter(y=history.history["val_loss"],name="Validation Loss")

data = [trace1,trace2]
py_offline.plot(data, filename='basic-line', include_plotlyjs=False, output_type='div')
```
<div>                            <div id="9a253f0c-d785-4a32-8fb4-40fba092d1ae" class="plotly-graph-div" style="height:100%; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("9a253f0c-d785-4a32-8fb4-40fba092d1ae")) {                    Plotly.newPlot(                        "9a253f0c-d785-4a32-8fb4-40fba092d1ae",                        [{"name":"Training loss","type":"scatter","y":[0.058278534561395645,0.05722469091415405,0.055976081639528275,0.05413749814033508,0.05195949599146843,0.04983977600932121,0.047649431973695755,0.045335717499256134,0.042959924787282944,0.04062921181321144,0.03834312781691551,0.03613226115703583,0.034103430807590485,0.032226432114839554,0.030500391498208046,0.028956780210137367,0.027625905349850655,0.026466840878129005,0.02543344534933567,0.024553125724196434,0.023776769638061523,0.023114116862416267,0.022525107488036156,0.02204899489879608,0.021653564646840096,0.021222712472081184,0.020815474912524223,0.02050328254699707,0.020228561013936996,0.01997336931526661,0.019709087908267975,0.019487522542476654,0.019291188567876816,0.019120845943689346,0.018972262740135193,0.01884193904697895,0.01869540847837925,0.018570875748991966,0.018489936366677284,0.018385285511612892]},{"name":"Validation Loss","type":"scatter","y":[0.05377873405814171,0.05285775288939476,0.05161994323134422,0.050348639488220215,0.049337007105350494,0.04812619462609291,0.04682121425867081,0.04544347524642944,0.04408815875649452,0.04306970164179802,0.04215961694717407,0.041414909064769745,0.040336973965168,0.03945089876651764,0.03880546614527702,0.0382448211312294,0.037505149841308594,0.0367407463490963,0.03622050955891609,0.035524141043424606,0.035079870373010635,0.034299202263355255,0.03374018892645836,0.033646248281002045,0.03341890498995781,0.032838888466358185,0.032575566321611404,0.032258376479148865,0.032065022736787796,0.03204252943396568,0.032129280269145966,0.032059311866760254,0.03175316005945206,0.03152251988649368,0.031246867030858994,0.03106059320271015,0.03096027299761772,0.031274210661649704,0.031148653477430344,0.03088766522705555]}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}}},                        {"responsive": true}                    )                };                            </script>        </div>



## Reconstruction error

We now have a model that can encode and decode ECG. Let's use the model to reconstruct a particular ECG and check the reconstruction error, i.e., the difference between the actual ECG and its reconstruction. 



```python
encoded_data = autoencoder.encoder(normal_test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()


trace1 = go.Scatter(y=normal_test_data[0],name='Input data')
trace2 = go.Scatter(y=decoded_data[0],name='Reconstruction & error',
                         fill='tonexty')

data = [trace1,trace2]
py_offline.plot(data, filename='basic-line', include_plotlyjs=False, output_type='div')

```
<div>                            <div id="dcc55352-bca6-4c4c-aac4-ad4200bf7f8a" class="plotly-graph-div" style="height:100%; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("dcc55352-bca6-4c4c-aac4-ad4200bf7f8a")) {                    Plotly.newPlot(                        "dcc55352-bca6-4c4c-aac4-ad4200bf7f8a",                        [{"name":"Input data","type":"scatter","y":[0.48035767674446106,0.2887779176235199,0.19828546047210693,0.17403002083301544,0.19065187871456146,0.2570231258869171,0.35133999586105347,0.3795180916786194,0.41933903098106384,0.47931399941444397,0.5085671544075012,0.5021305084228516,0.4981786012649536,0.5009568333625793,0.4963546097278595,0.4957442283630371,0.49634596705436707,0.4871847927570343,0.49864014983177185,0.49027618765830994,0.48405030369758606,0.4853919446468353,0.4781576991081238,0.4776414632797241,0.4754825830459595,0.4680434465408325,0.4633431136608124,0.4667298495769501,0.4565943777561188,0.45639845728874207,0.44269412755966187,0.4387221038341522,0.4350164234638214,0.4311322867870331,0.43271467089653015,0.43053433299064636,0.4260518252849579,0.4177199602127075,0.4224426746368408,0.4249061048030853,0.4222816526889801,0.4255194664001465,0.427318811416626,0.4344993233680725,0.43405434489250183,0.4319981336593628,0.4433560073375702,0.4449458420276642,0.4500327408313751,0.44775307178497314,0.45146772265434265,0.4631691873073578,0.4599079489707947,0.4583589732646942,0.46505701541900635,0.4695504903793335,0.48476895689964294,0.4720330536365509,0.47590193152427673,0.48128950595855713,0.48106202483177185,0.483425110578537,0.4797944724559784,0.49654191732406616,0.497626930475235,0.5003345012664795,0.4999160170555115,0.4998267590999603,0.5085312724113464,0.5027685165405273,0.5192650556564331,0.5174381136894226,0.5176679491996765,0.5243684649467468,0.522845447063446,0.523930549621582,0.5308157205581665,0.5383780002593994,0.5319925546646118,0.5184898376464844,0.5222746729850769,0.5139047503471375,0.5156378149986267,0.5147913098335266,0.5096921324729919,0.5125722289085388,0.5018595457077026,0.49053633213043213,0.48884034156799316,0.4823535084724426,0.47934600710868835,0.48330995440483093,0.4766990542411804,0.47691449522972107,0.4795263111591339,0.4947693347930908,0.5108534097671509,0.5188336968421936,0.5332105159759521,0.5425611138343811,0.5682035684585571,0.5826812386512756,0.6099961996078491,0.6065466403961182,0.5954951047897339,0.5819999575614929,0.5612176060676575,0.5507384538650513,0.5170336961746216,0.4897701144218445,0.4583636224269867,0.41286134719848633,0.3860044479370117,0.3743003010749817,0.37155696749687195,0.36838045716285706,0.3640936613082886,0.3643512427806854,0.3664138913154602,0.3565950393676758,0.3636854290962219,0.3610741198062897,0.3605392873287201,0.3603725731372833,0.36098870635032654,0.3649810552597046,0.3636249601840973,0.36424019932746887,0.3778349459171295,0.4150768220424652,0.4684041142463684,0.46837979555130005,0.48350921273231506,0.4904227554798126,0.4625466465950012,0.43603575229644775,0.4314790964126587,0.45463305711746216,0.5246124863624573,0.37137290835380554]},{"fill":"tonexty","name":"Reconstruction & error","type":"scatter","y":[0.4465961456298828,0.3107976019382477,0.21133604645729065,0.18289929628372192,0.18875935673713684,0.2345043122768402,0.3434072732925415,0.3885255455970764,0.36512336134910583,0.42534133791923523,0.4633747935295105,0.4717438817024231,0.44047656655311584,0.4607338607311249,0.4668833911418915,0.46644940972328186,0.48185503482818604,0.46478328108787537,0.4626923203468323,0.4700751006603241,0.46916255354881287,0.4654800295829773,0.4807877540588379,0.46511152386665344,0.46649792790412903,0.4589080512523651,0.43502575159072876,0.45252928137779236,0.4419180750846863,0.4393032193183899,0.42319244146347046,0.44804516434669495,0.4537696838378906,0.4307486116886139,0.4432544410228729,0.43226516246795654,0.438921183347702,0.41973650455474854,0.4481489062309265,0.4237235188484192,0.4533311128616333,0.4676003158092499,0.45611345767974854,0.44850221276283264,0.4738155007362366,0.47608277201652527,0.4773779809474945,0.47266486287117004,0.4874640107154846,0.4830182194709778,0.482825368642807,0.4895108938217163,0.47653865814208984,0.4739570915699005,0.4788680672645569,0.46779096126556396,0.4797886908054352,0.4860452711582184,0.48060664534568787,0.4844375252723694,0.47528624534606934,0.4678887724876404,0.47648727893829346,0.4845319986343384,0.4717547297477722,0.4927765130996704,0.48250359296798706,0.4789403975009918,0.4808230400085449,0.49723130464553833,0.49220043420791626,0.49434182047843933,0.48986905813217163,0.4954819083213806,0.4968757927417755,0.4968879520893097,0.49078139662742615,0.5026766657829285,0.4936281740665436,0.4938465654850006,0.4941617548465729,0.4906451404094696,0.4884297549724579,0.4897095263004303,0.48889023065567017,0.48968732357025146,0.48284363746643066,0.4703805446624756,0.48027828335762024,0.4846431612968445,0.47830653190612793,0.47400787472724915,0.47063881158828735,0.4844090938568115,0.48908936977386475,0.49187833070755005,0.5100383758544922,0.5182695388793945,0.5214661359786987,0.5442376136779785,0.553776741027832,0.5636242628097534,0.570266604423523,0.5791199207305908,0.5670091509819031,0.5490534901618958,0.5640684366226196,0.5354028344154358,0.5078877210617065,0.49572017788887024,0.47165119647979736,0.46608033776283264,0.47511523962020874,0.43220263719558716,0.4526681900024414,0.4415639340877533,0.44471877813339233,0.453139990568161,0.4496006667613983,0.42963409423828125,0.4544813632965088,0.43883538246154785,0.41174355149269104,0.4584041237831116,0.42554694414138794,0.45842498540878296,0.4576203227043152,0.44085976481437683,0.43572020530700684,0.4835623800754547,0.514413058757782,0.5381374359130859,0.5366976857185364,0.5121092796325684,0.5094010829925537,0.45953381061553955,0.4980086386203766,0.4697421193122864,0.47742846608161926,0.3997027277946472]}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}}},                        {"responsive": true}                    )                };                            </script>        </div>



Let's do the same as above for an abnormal rhythm ECG on the test dataset.


```python
encoded_data = autoencoder.encoder(abnormal_test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()


trace1 = go.Scatter(y=abnormal_test_data[0],name='Input data')
trace2 = go.Scatter(y=decoded_data[0],name='Reconstruction & error',
                         fill='tonexty')


data = [trace1,trace2]
py_offline.plot(data, filename='basic-line', include_plotlyjs=False, output_type='div')
```
<div>                            <div id="aa16bf73-cc26-4abf-ab3a-fac78ae700e0" class="plotly-graph-div" style="height:100%; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("aa16bf73-cc26-4abf-ab3a-fac78ae700e0")) {                    Plotly.newPlot(                        "aa16bf73-cc26-4abf-ab3a-fac78ae700e0",                        [{"name":"Input data","type":"scatter","y":[0.3687897026538849,0.30728116631507874,0.2658798396587372,0.2342016100883484,0.2289544939994812,0.24441950023174286,0.2897324562072754,0.32673344016075134,0.3450625240802765,0.35946375131607056,0.366017609834671,0.38083764910697937,0.4098110496997833,0.426404744386673,0.4301389157772064,0.42872142791748047,0.4324492812156677,0.42882904410362244,0.4317876100540161,0.4314000606536865,0.4309997260570526,0.43363940715789795,0.43189460039138794,0.4334389865398407,0.4282280206680298,0.4304920434951782,0.4316231906414032,0.43162646889686584,0.4328855872154236,0.42824631929397583,0.43387359380722046,0.4338427782058716,0.4270707368850708,0.4270402491092682,0.4258621335029602,0.42957353591918945,0.43204739689826965,0.42690134048461914,0.4267188310623169,0.42579153180122375,0.42145025730133057,0.42689019441604614,0.4192477762699127,0.42660969495773315,0.425228089094162,0.4170193672180176,0.4184109568595886,0.41961243748664856,0.41716280579566956,0.41829556226730347,0.41633373498916626,0.4121170938014984,0.4156325161457062,0.4140487611293793,0.4120148718357086,0.41335198283195496,0.4142370820045471,0.41685929894447327,0.41644006967544556,0.41624537110328674,0.4196259379386902,0.4177689850330353,0.41905897855758667,0.42218971252441406,0.4196544587612152,0.42940545082092285,0.4308989942073822,0.4307335913181305,0.4318483769893646,0.43540969491004944,0.44192975759506226,0.445151150226593,0.44695404171943665,0.4487036466598511,0.45294925570487976,0.4540736973285675,0.45773977041244507,0.4565097689628601,0.46157169342041016,0.4622882306575775,0.46499142050743103,0.46702536940574646,0.46528127789497375,0.4724016785621643,0.4684238135814667,0.4714818596839905,0.4761196970939636,0.4806874096393585,0.477120041847229,0.47579425573349,0.4818546175956726,0.48146358132362366,0.4816263020038605,0.4840543866157532,0.4852852523326874,0.48290780186653137,0.4826079308986664,0.48741355538368225,0.4890534281730652,0.49074432253837585,0.49717646837234497,0.4968552887439728,0.4969344139099121,0.5008605718612671,0.5031581521034241,0.5059995651245117,0.5076162815093994,0.5129083395004272,0.5130175948143005,0.5137396454811096,0.5173085331916809,0.5215231776237488,0.5224050283432007,0.5229285359382629,0.5217630863189697,0.5279528498649597,0.5248215794563293,0.5252198576927185,0.531267523765564,0.5323393940925598,0.5313363671302795,0.5249910354614258,0.531597912311554,0.5341436862945557,0.5397691130638123,0.5453839898109436,0.5644707083702087,0.5959868431091309,0.6098214387893677,0.6092298626899719,0.6011083722114563,0.5885306596755981,0.5727330446243286,0.5786002278327942,0.6149988770484924,0.6372957229614258,0.6170028448104858,0.5855123400688171,0.56629878282547,0.5726404190063477]},{"fill":"tonexty","name":"Reconstruction & error","type":"scatter","y":[0.4187658429145813,0.25643202662467957,0.17971700429916382,0.1416165828704834,0.14625221490859985,0.19912287592887878,0.28362882137298584,0.306428998708725,0.36866244673728943,0.40833917260169983,0.41588500142097473,0.41721493005752563,0.4349389970302582,0.4283873736858368,0.42799243330955505,0.42019379138946533,0.41183748841285706,0.4192914366722107,0.41870197653770447,0.4112783372402191,0.41204380989074707,0.41428256034851074,0.3987419009208679,0.40743571519851685,0.402587890625,0.40924328565597534,0.41743195056915283,0.3984314799308777,0.4067050516605377,0.40339693427085876,0.4046981930732727,0.38910001516342163,0.3745383322238922,0.38830700516700745,0.3803955614566803,0.38328105211257935,0.385261595249176,0.3983069062232971,0.38237297534942627,0.4034058451652527,0.3934061527252197,0.39021554589271545,0.4104977548122406,0.42409589886665344,0.41617658734321594,0.4212360382080078,0.4310763478279114,0.43970581889152527,0.43412071466445923,0.4380657970905304,0.44638338685035706,0.4434937834739685,0.44985634088516235,0.4562239944934845,0.45253682136535645,0.4610736072063446,0.45664986968040466,0.4536914527416229,0.45391586422920227,0.45723089575767517,0.4618476331233978,0.47055482864379883,0.46413254737854004,0.46391597390174866,0.47588634490966797,0.4670261740684509,0.4762842059135437,0.4825576841831207,0.4902157187461853,0.4868711531162262,0.49219056963920593,0.49171653389930725,0.4993748664855957,0.5015336275100708,0.5029653906822205,0.5024036169052124,0.5100195407867432,0.49737900495529175,0.5062448978424072,0.5006082653999329,0.4974426031112671,0.49929752945899963,0.5006028413772583,0.49739477038383484,0.50086510181427,0.49359241127967834,0.4955478608608246,0.5003208518028259,0.4959305226802826,0.49367836117744446,0.49034953117370605,0.49589961767196655,0.49691149592399597,0.49115365743637085,0.4901222288608551,0.49491578340530396,0.4970593750476837,0.5090397596359253,0.5271615386009216,0.5299252867698669,0.547936201095581,0.5641404986381531,0.5785156488418579,0.5862573981285095,0.5922874212265015,0.5971267819404602,0.5787860751152039,0.5738573670387268,0.5701033473014832,0.5513163805007935,0.5333587527275085,0.5101572871208191,0.461076021194458,0.4704042673110962,0.44186559319496155,0.4508987367153168,0.4458563029766083,0.4369012117385864,0.4401966333389282,0.4476768672466278,0.43786871433258057,0.4420301616191864,0.46228861808776855,0.43606051802635193,0.45663702487945557,0.43588000535964966,0.43653735518455505,0.46380069851875305,0.48262155055999756,0.5013130903244019,0.5355772376060486,0.5378100872039795,0.5445823073387146,0.5611144304275513,0.5552237033843994,0.5626068711280823,0.5326967239379883,0.5376222729682922,0.5379247665405273,0.4466624855995178]}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}}},                        {"responsive": true}                    )                };                            </script>        </div>



With the naked eye, the two plots above seem to suggest that the reconstruction error for the abnormal rhythm ECG is larger. We will formalize our findings in the next section.

## Detecting anomalies

Here we will compute the reconstruction error for all the data points both normal and abnormal.  For the reconstruction error, we will use the mean absolute error.
We will compute the reconstruction error of the training dataset and choose a threshold  (one standard deviation away from the mean) above which we will classify an ECG as abnormal.



```python
reconstructions = autoencoder.predict(normal_train_data)
train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)

fig = go.Figure()
fig = fig.add_trace(go.Histogram(x=train_loss[None,:][0],name='Normal loss'))
fig.show()
```
![Caption of here](../../images/error2.png){:height="100%" width="100%"}

We now define the threshold.


```python
threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold)
```

    Threshold:  0.028107813


On the test dataset, we will use the threshold above to determine anormalies. We will do this as follows:


```python
reconstructions_normal = autoencoder.predict(normal_test_data)
test_loss_normal = tf.keras.losses.mae(reconstructions_normal, normal_test_data)


reconstructions_abnormal = autoencoder.predict(abnormal_test_data)
test_loss_abnormal = tf.keras.losses.mae(reconstructions_abnormal, abnormal_test_data)

fig = go.Figure()
fig.add_trace(go.Histogram(x=test_loss_normal[None,:][0],name='Normal loss'))
fig.add_trace(go.Histogram(x=test_loss_abnormal[None,:][0],name='Abnormal loss',opacity=0.4))

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



![Caption of here](../../images/error.png){:height="100%" width="100%"}




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

    Accuracy = 0.939
    Precision = 0.994059405940594
    Recall = 0.8964285714285715


## Final words

In this blog post, we have seen how autoencoders can be used to detect anomalies in our data. The ECG data is a  nice example to illustrate the idea, however, with a typical real-world use case, there will be more shortcomings. 
