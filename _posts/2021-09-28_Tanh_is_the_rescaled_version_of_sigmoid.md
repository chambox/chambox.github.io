---
layout: post
title: Tanh is the rescaled version of the sigmoid.
---

<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

Deep learning today uses different activation functions. In this short blog post I will show you the relationship between the `tanh` and the `sigmoid` activation function. In fact I will show that the `tanh` is just a rescaled version of the `sigmoid`. 

$$
\text{tanh}(x) = \frac{e^{2x}-1}{e^{2x}+1}
$$

$$
\text{sigmoid}(x) = \frac{e^x}{e^x+1}
$$

Now look 
$$
\begin{align*}
2 \text{sigmoid}(2x) -1& = 2\frac{e^{2x}}{1+e^{2x}}-1\\
&= \frac{e^{2x}}{1+e^{2x}}+\frac{e^{2x}}{1+e^[2x]}-\frac{1+e^{2x}}{1+e^{2x}}\\
&=  \frac{e^{2x}+e^{2x}-1-e^{2x}}{1+e^{2x}}\\
&= \frac{e^{2x}-1}{1+e^{2x}} = \frac{e^{2x}-1}{e^{2x}+1} = \text{tanh}(x)
\end{align*}
$$


Hence we can safely say the `sigmoid` is rescaled version of the `tanh`. 



