---
layout: post
title: Anomaly detection with autoencoders (10 mins read) 
---


<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>


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
import plotly.offline as py_offline
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
trace=go.Scatter(x=np.arange(140), y=normal_train_data[0],name='Normal')
trace2=go.Scatter(x=np.arange(140), y=abnormal_train_data[0],name='Abnormal')
data = [trace,trace2]
py_offline.plot(data, filename='basic-line', include_plotlyjs=True, output_type='div')


```




    s=r.target,f=i.length;e._length&&(f=Math.min(f,e._length));var h=r.targetcalendar,p=e._arrayAttrs,d=r.preservegaps;if("string"==typeof s){var m=n.nestedProperty(e,s+"calendar").get();m&&(h=m)}var g,v,y=function(t,e,r){var n=t.operation,i=t.value,a=Array.isArray(i);function o(t){return-1!==t.indexOf(n)}var s,f=function(r){return e(r,0,t.valuecalendar)},h=function(t){return e(t,0,r)};o(l)?s=f(a?i[0]:i):o(c)?s=a?[f(i[0]),f(i[1])]:[f(i),f(i)]:o(u)&&(s=a?i.map(f):[f(i)]);switch(n){case"=":return function(t){return h(t)===s};case"!=":return function(t){return h(t)!==s};case"<":return function(t){return h(t)<s};case"<=":return function(t){return h(t)<=s};case">":return function(t){return h(t)>s};case">=":return function(t){return h(t)>=s};case"[]":return function(t){var e=h(t);return e>=s[0]&&e<=s[1]};case"()":return function(t){var e=h(t);return e>s[0]&&e<s[1]};case"[)":return function(t){var e=h(t);return e>=s[0]&&e<s[1]};case"(]":return function(t){var e=h(t);return e>s[0]&&e<=s[1]};case"][":return function(t){var e=h(t);return e<=s[0]||e>=s[1]};case")(":return function(t){var e=h(t);return e<s[0]||e>s[1]};case"](":return function(t){var e=h(t);return e<=s[0]||e>s[1]};case")[":return function(t){var e=h(t);return e<s[0]||e>=s[1]};case"{}":return function(t){return-1!==s.indexOf(h(t))};case"}{":return function(t){return-1===s.indexOf(h(t))}}}(r,a.getDataToCoordFunc(t,e,s,i),h),x={},b={},_=0;d?(g=function(t){x[t.astr]=n.extendDeep([],t.get()),t.set(new Array(f))},v=function(t,e){var r=x[t.astr][e];t.get()[e]=r}):(g=function(t){x[t.astr]=n.extendDeep([],t.get()),t.set([])},v=function(t,e){var r=x[t.astr][e];t.get().push(r)}),k(g);for(var w=o(e.transforms,r),T=0;T<f;T++){y(i[T])?(k(v,T),b[_++]=w(T)):d&&_++}r._indexToPoints=b,e._length=_}}function k(t,r){for(var i=0;i<p.length;i++){t(n.nestedProperty(e,p[i]),r)}}}},{"../constants/filter_ops":767,"../lib":795,"../plots/cartesian/axes":845,"../registry":923,"./helpers":1394}],1393:[function(t,e,r){"use strict";var n=t("../lib"),i=t("../plot_api/plot_schema"),a=t("../plots/plots"),o=t("./helpers").pointsAccessorFunction;function s(t,e){var r,s,l,c,u,f,h,p,d,m,g=e.transform,v=e.transformIndex,y=t.transforms[v].groups,x=o(t.transforms,g);if(!n.isArrayOrTypedArray(y)||0===y.length)return[t];var b=n.filterUnique(y),_=new Array(b.length),w=y.length,T=i.findArrayAttributes(t),k=g.styles||[],M={};for(r=0;r<k.length;r++)M[k[r].target]=k[r].value;g.styles&&(m=n.keyedContainer(g,"styles","target","value.name"));var A={},S={};for(r=0;r<b.length;r++){A[f=b[r]]=r,S[f]=0,(h=_[r]=n.extendDeepNoArrays({},t))._group=f,h.transforms[v]._indexToPoints={};var E=null;for(m&&(E=m.get(f)),h.name=E||""===E?E:n.templateString(g.nameformat,{trace:t.name,group:f}),p=h.transforms,h.transforms=[],s=0;s<p.length;s++)h.transforms[s]=n.extendDeepNoArrays({},p[s]);for(s=0;s<T.length;s++)n.nestedProperty(h,T[s]).set([])}for(l=0;l<T.length;l++){for(c=T[l],s=0,d=[];s<b.length;s++)d[s]=n.nestedProperty(_[s],c).get();for(u=n.nestedProperty(t,c).get(),s=0;s<w;s++)d[A[y[s]]].push(u[s])}for(s=0;s<w;s++){(h=_[A[y[s]]]).transforms[v]._indexToPoints[S[y[s]]]=x(s),S[y[s]]++}for(r=0;r<b.length;r++)f=b[r],h=_[r],a.clearExpandedTraceDefaultColors(h),h=n.extendDeepNoArrays(h,M[f]||{});return _}r.moduleType="transform",r.name="groupby",r.attributes={enabled:{valType:"boolean",dflt:!0,editType:"calc"},groups:{valType:"data_array",dflt:[],editType:"calc"},nameformat:{valType:"string",editType:"calc"},styles:{_isLinkedToArray:"style",target:{valType:"string",editType:"calc"},value:{valType:"any",dflt:{},editType:"calc",_compareAsJSON:!0},editType:"calc"},editType:"calc"},r.supplyDefaults=function(t,e,i){var a,o={};function s(e,i){return n.coerce(t,o,r.attributes,e,i)}if(!s("enabled"))return o;s("groups"),s("nameformat",i._dataLength>1?"%{group} (%{trace})":"%{group}");var l=t.styles,c=o.styles=[];if(l)for(a=0;a<l.length;a++){var u=c[a]={};n.coerce(l[a],c[a],r.attributes.styles,"target");var f=n.coerce(l[a],c[a],r.attributes.styles,"value");n.isPlainObject(f)?u.value=n.extendDeep({},f):f&&delete u.value}return o},r.transform=function(t,e){var r,n,i,a=[];for(n=0;n<t.length;n++)for(r=s(t[n],e),i=0;i<r.length;i++)a.push(r[i]);return a}},{"../lib":795,"../plot_api/plot_schema":833,"../plots/plots":909,"./helpers":1394}],1394:[function(t,e,r){"use strict";r.pointsAccessorFunction=function(t,e){for(var r,n,i=0;i<t.length&&(r=t[i])!==e;i++)r._indexToPoints&&!1!==r.enabled&&(n=r._indexToPoints);return n?function(t){return n[t]}:function(t){return[t]}}},{}],1395:[function(t,e,r){"use strict";var n=t("../lib"),i=t("../plots/cartesian/axes"),a=t("./helpers").pointsAccessorFunction,o=t("../constants/numerical").BADNUM;r.moduleType="transform",r.name="sort",r.attributes={enabled:{valType:"boolean",dflt:!0,editType:"calc"},target:{valType:"string",strict:!0,noBlank:!0,arrayOk:!0,dflt:"x",editType:"calc"},order:{valType:"enumerated",values:["ascending","descending"],dflt:"ascending",editType:"calc"},editType:"calc"},r.supplyDefaults=function(t){var e={};function i(i,a){return n.coerce(t,e,r.attributes,i,a)}return i("enabled")&&(i("target"),i("order")),e},r.calcTransform=function(t,e,r){if(r.enabled){var s=n.getTargetArray(e,r);if(s){var l=r.target,c=s.length;e._length&&(c=Math.min(c,e._length));var u,f,h=e._arrayAttrs,p=function(t,e,r,n){var i,a=new Array(n),s=new Array(n);for(i=0;i<n;i++)a[i]={v:e[i],i:i};for(a.sort(function(t,e){switch(t.order){case"ascending":return function(t,r){var n=e(t.v),i=e(r.v);return n===o?1:i===o?-1:n-i};case"descending":return function(t,r){var n=e(t.v),i=e(r.v);return n===o?1:i===o?-1:i-n}}}(t,r)),i=0;i<n;i++)s[i]=a[i].i;return s}(r,s,i.getDataToCoordFunc(t,e,l,s),c),d=a(e.transforms,r),m={};for(u=0;u<h.length;u++){var g=n.nestedProperty(e,h[u]),v=g.get(),y=new Array(c);for(f=0;f<c;f++)y[f]=v[p[f]];g.set(y)}for(f=0;f<c;f++)m[f]=d(p[f]);r._indexToPoints=m,e._length=c}}}},{"../constants/numerical":771,"../lib":795,"../plots/cartesian/axes":845,"./helpers":1394}],1396:[function(t,e,r){"use strict";r.version="2.1.0"},{}]},{},[27])(27)}));</script>                <div id="1cb531bd-db68-4129-8fba-c3b07b2045bd" class="plotly-graph-div" style="height:100%; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("1cb531bd-db68-4129-8fba-c3b07b2045bd")) {                    Plotly.newPlot(                        "1cb531bd-db68-4129-8fba-c3b07b2045bd",                        [{"name":"Normal","type":"scatter","x":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139],"y":[0.5703046321868896,0.4656165838241577,0.2905811667442322,0.17791584134101868,0.09538919478654861,0.08467857539653778,0.2019510418176651,0.3163002133369446,0.33732032775878906,0.41424882411956787,0.47070595622062683,0.4691905081272125,0.4776775538921356,0.48004090785980225,0.4702724516391754,0.4729926884174347,0.479171484708786,0.48027467727661133,0.46928870677948,0.46106863021850586,0.46148037910461426,0.4465829133987427,0.45325326919555664,0.4561009407043457,0.4489617943763733,0.44325318932533264,0.4340217411518097,0.4430723488330841,0.4325052499771118,0.43014901876449585,0.41454657912254333,0.4120652973651886,0.40420278906822205,0.4105454385280609,0.4082913100719452,0.40342992544174194,0.39648476243019104,0.3933204412460327,0.39158815145492554,0.3960387706756592,0.4074695408344269,0.40531179308891296,0.41734880208969116,0.41624557971954346,0.42334118485450745,0.4459063708782196,0.4455184042453766,0.44474098086357117,0.44404327869415283,0.4567321836948395,0.4521746337413788,0.4532504081726074,0.46082931756973267,0.4669593572616577,0.4663194417953491,0.47432830929756165,0.46297091245651245,0.4738894999027252,0.4678889811038971,0.4673447906970978,0.4802667498588562,0.48439839482307434,0.48574668169021606,0.48985567688941956,0.4931734800338745,0.48830410838127136,0.49913936853408813,0.505709171295166,0.5078241229057312,0.5132302641868591,0.5211992859840393,0.514991819858551,0.519900918006897,0.5074853897094727,0.5113434195518494,0.5092939734458923,0.5073481798171997,0.5110538005828857,0.5096123814582825,0.4963955879211426,0.49456536769866943,0.5014910101890564,0.5063670873641968,0.5020293593406677,0.5007152557373047,0.4958237409591675,0.4843623638153076,0.4914246201515198,0.4825357496738434,0.47723883390426636,0.47550636529922485,0.47530031204223633,0.4876338243484497,0.4776129722595215,0.474583238363266,0.4757544696331024,0.47009119391441345,0.4540967047214508,0.46874862909317017,0.4767007529735565,0.4795878529548645,0.4751480519771576,0.47619494795799255,0.48561891913414,0.48763763904571533,0.4996277689933777,0.5284430384635925,0.5431970953941345,0.5495933294296265,0.5488370060920715,0.5231477618217468,0.4934438467025757,0.49881476163864136,0.5099680423736572,0.516133189201355,0.4907889664173126,0.4668024778366089,0.43299439549446106,0.4164400100708008,0.4182245135307312,0.43140411376953125,0.43214115500450134,0.4212411642074585,0.42373567819595337,0.42885276675224304,0.43089887499809265,0.43537637591362,0.4391244351863861,0.4371418356895447,0.4453428089618683,0.4533092975616455,0.48821336030960083,0.5786804556846619,0.5858615636825562,0.5959517955780029,0.5952476263046265,0.5700759291648865,0.4850423336029053,0.42335018515586853,0.4759834408760071]},{"name":"Abnormal","type":"scatter","x":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139],"y":[0.4304001033306122,0.35345321893692017,0.3034263849258423,0.2818489074707031,0.28353944420814514,0.28959953784942627,0.3114522695541382,0.3470645248889923,0.3809654414653778,0.39506226778030396,0.3976452350616455,0.41356122493743896,0.4387162923812866,0.4520491063594818,0.4495706260204315,0.44956129789352417,0.45499545335769653,0.45426806807518005,0.44863343238830566,0.45084789395332336,0.45330581068992615,0.4520796537399292,0.4475529193878174,0.44939249753952026,0.4473649561405182,0.44874852895736694,0.4485017955303192,0.4462437331676483,0.44809791445732117,0.4483480751514435,0.44390979409217834,0.44208911061286926,0.4433857798576355,0.44462850689888,0.43986889719963074,0.4404641091823578,0.4400186240673065,0.43668332695961,0.4339315891265869,0.4317610263824463,0.4295136630535126,0.4274426996707916,0.42877644300460815,0.42894187569618225,0.4254743754863739,0.42517906427383423,0.42631953954696655,0.4277213215827942,0.42250746488571167,0.42367860674858093,0.4281102418899536,0.4270593523979187,0.4272007346153259,0.4308776557445526,0.4309622049331665,0.43490204215049744,0.43578556180000305,0.43896228075027466,0.44403478503227234,0.4491182863712311,0.453519344329834,0.4524179995059967,0.45339763164520264,0.45347079634666443,0.45640361309051514,0.45917513966560364,0.45685112476348877,0.45898863673210144,0.4569636881351471,0.45675793290138245,0.46130383014678955,0.45900118350982666,0.46151021122932434,0.46561598777770996,0.4587060809135437,0.4582265317440033,0.4626377522945404,0.4621920883655548,0.4627910554409027,0.46581798791885376,0.4644567370414734,0.4647052586078644,0.46456974744796753,0.4647565186023712,0.46593987941741943,0.46568334102630615,0.46936455368995667,0.47205060720443726,0.4691234230995178,0.4770551025867462,0.4743116497993469,0.4712502956390381,0.476502001285553,0.4755341112613678,0.47373101115226746,0.47748467326164246,0.48040735721588135,0.47647133469581604,0.4788639545440674,0.48452094197273254,0.4846736788749695,0.4795249104499817,0.4818657338619232,0.4819636642932892,0.48308029770851135,0.47801387310028076,0.48302972316741943,0.48857638239860535,0.48736441135406494,0.49351075291633606,0.49945521354675293,0.505765438079834,0.5169975757598877,0.527177631855011,0.5347791314125061,0.544451892375946,0.5556207895278931,0.5749051570892334,0.5851321816444397,0.5897799730300903,0.5978419184684753,0.6050950884819031,0.6238124370574951,0.6344885230064392,0.6365599036216736,0.6482247710227966,0.6608508229255676,0.6543368697166443,0.6259334683418274,0.5985985398292542,0.5772880911827087,0.5069698095321655,0.423368364572525,0.395474374294281,0.3666984438896179,0.35847392678260803,0.34862595796585083,0.3050689399242401,0.2622550129890442,0.25020110607147217]}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}}},                        {"responsive": true}                    )                };                            </script>        </div>'



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

    2021-07-01 15:42:58.525013: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:176] 
    None of the MLIR Optimization Passes are enabled (registered 2)
    2021-07-01 15:42:58.525661: I tensorflow/core/platform/profile_utils/cpu_utils.cc:114] CPU Frequency: 2299965000 Hz
    Epoch 1/40
    3/3 [==============================] - 1s 109ms/step - loss: 0.0574 - val_loss: 0.0530
    Epoch 2/40
    3/3 [==============================] - 0s 13ms/step - loss: 0.0557 - val_loss: 0.0518
    Epoch 3/40
    3/3 [==============================] - 0s 31ms/step - loss: 0.0538 - val_loss: 0.0507
    Epoch 4/40
    3/3 [==============================] - 0s 23ms/step - loss: 0.0518 - val_loss: 0.0497
    Epoch 5/40
    3/3 [==============================] - 0s 18ms/step - loss: 0.0497 - val_loss: 0.0486
    Epoch 6/40
    3/3 [==============================] - 0s 16ms/step - loss: 0.0475 - val_loss: 0.0475
    Epoch 7/40
    3/3 [==============================] - 0s 11ms/step - loss: 0.0452 - val_loss: 0.0464



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

Threshold:  0.0304742


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

    Accuracy = 0.942
    Precision = 0.9940944881889764
    Recall = 0.9017857142857143


## Final words

In this blog post, we have seen how autoencoders can be used to detect anomalies in our data. The ECG data is a  nice example to illustrate the idea, however, with a typical real-world use case, there will be more shortcomings. 
