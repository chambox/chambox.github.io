```python
print('Hellow')
```

    Hellow



```python
import pandas as pd 
import numpy as np 
import plotly.offline as py_offline
import plotly.graph_objs as go
```


```python
N = 10
random_x = np.linspace(0, 1, N)
random_y = np.random.randn(N)

trace = go.Scatter(
    x = random_x,
    y = random_y
)

data = [trace]
```


```python
import plotly.graph_objects as go
fig = go.Figure(
    data=[go.Bar(y=[2, 1, 3])],
    layout_title_text="A Figure Displaying Itself"
)
fig.show()
```




```python

```
