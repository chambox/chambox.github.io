---
layout: post
title: Regression to the mean and James Stein estimators
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

The title, “regression to the mean”, immediately sounds sophisticated, and may seem like an article exclusive to statisticians.  But when I read the international bestseller’s book:  [Thinking Fast and Slow](https://www.amazon.fr/Thinking-Fast-Slow-Daniel-Kahneman/dp/0141033576), by [Daniel Kahneman](https://en.wikipedia.org/wiki/Daniel_Kahneman), I noticed this is not true.  Daniel  Kahneman gives lots of excellent examples of regression to the mean.  Amongst them, I admired one, which I describe subsequently in this post.

Daniel  Kahneman starts his Chapter 17 describing a talk he gave to flight instructors in Israel.  In his speech, he emphasised the well-known principle of reward for improved performance rather than punishment for mistakes.   As a response to his talk, one of the instructors, raised his hand supposedly to ask a question but ended up giving a speech of his own. This instructor’s point of view was the opposite. He said when I train my flight cadets; usually, when I praised them for improved performance, it is followed by a bad performance when they are asked to do it again.  Conversely, when I shout in their earpiece for a lousy manoeuvre, the next one is usually better.

## Regression to the mean 

Daniel felt this was a eureka moment to discuss regression to the mean, it was so crucial to him that he even wrote about it in his book.  Daniel in response to the instructor’s comment said this: “As an instructor, you have probably seen/observed many manoeuvres made by your flight cadets.   And most probably you only praised flight cadets, if their manoeuvres were far better than the average performance. The next time they try to do the same manoeuvre, they do worse just because there was a little element of chance in the first manoeuvre, making it extremely good.  On the other hand, the instructor probably only shouts in the earphones of a flight cadet, if his manoeuvre is far worse than the average performance of other flight cadets.  The next time he tries to do the same manoeuvre, he does better, just because in the first manoeuvre he had an extreme bad luck”.   This means flight cadets performances turn to stay around the average performance. If a flight cadet will repeat their manoeuvre the second time when they had an extreme performance in the first, the second performance turns to go in the opposite direction of the first. The second time, outstanding performances go worse and terrible performances get better.  Also, most likely, flight cadets that stay consistent in their performance, first and second, are those whose performances are close to the average performance.   

### Causal effect

The flight instructor’s comment immediately attributed a causal effect of praise or shouting to the earphones of the flight cadets, to their performance in the next flight manoeuvre. However, there is no link between praise or shouting to flight cadets’ earphones and their second performances. If the flight instructor would have observed very bad or very good performances and did not react, he would have realised this.  Him reacting to or not reacting to the flight cadet's performances will lead to the same outcome.  It is a natural phenomenon that flight cadets manoeuvre performance turns to stay around the average performance of the group.   This phenomenon is known as regression to the mean; it has no cause, it is just a simple law that governs the flight cadets manoeuvre performances.  Statisticians call this the Gaussian law, a bell-shaped symmetric distribution.


## Origin 
[Sir Francis Galton](https://en.wikipedia.org/wiki/Francis_Galton) was the first to illustrate regression to the mean. In one of his famous examples, he showed that large seeds produce offspring seeds that are less as large. And tiny seeds, on the other hand, turn to have offspring seeds that are larger. This is another example of regression to the mean.  

## Formal presentation


In a more formal way,  lets start with  $$n$$ observations labelled as $$x_i$$, $$i=1,\ldots,n$$. Further we assume that, each  $$x_i$$ (independently) is govern by the Gaussian law, that is $$x_i\sim N(\mu_i,1)$$, where $$\mu_i$$ is the mean, and $$1$$ is the variation around this mean.  

Back to the flight cadet's manoeuvre example. We observed their first performance in a manoeuvre, $$x_i$$,  and wish to estimate their actual performance $$\mu_i$$. 


[Sir Ronald Aylmer Fisher](https://en.wikipedia.org/wiki/Ronald_Fisher) was an English statistician and biologist who used mathematics to combine Mendelian genetics and natural selection; he almost developed a complete philosophy (Fisherian statistics) on his own.  Fisher suggested that we estimate a cadet's actual manoeuvre performance by simply using $$x_i$$, his observed manoeuvre performance. He called his estimator, the maximum likelihood estimator, MLE. That is,

$$
 \hat \mu_i^\text{MLE}=x_i.
$$

After reading Daniel Kahneman’s example above on regression to the mean, we are all surprised why Fisher did not see that $$ \hat \mu_i^\text{MLE}$$ is a poor estimator of $$\mu_i$$.  Fortunately for us, [Charles Stein](https://en.wikipedia.org/wiki/Charles_M._Stein) later realised that $$ \hat \mu_i^\text{MLE}$$ is a poor estimator of $$\mu_i$$.

## Stein's paradox

A phenomenon known as Stein’s paradox, states: when three or more parameters (e.g., cadets true manoeuvre performances $$\mu_i$$, $$i=1,\ldots,n$$) are estimated simultaneously, there exist a combined estimator more accurate on average (that is, having lower expected mean squared error) than any method that handles the parameters separately. It is named after Charles Stein of Stanford University, who discovered the phenomenon in 1955.


Stein's paradox as stated does not immediately tell you anything about regression to the mean.
Six years later, in 1961, James and Stein actually worked out the James Stein estimator. It was not just an existence of such an estimator; an algorithm was given on how to construct the estimator. Now consider $$x=(x_1,\ldots,x_n)$$, and that $$\bar x=\sum x_i/n$$ (the mean), the James Stein estimator has the form,


$$
 \hat \mu_i^\text{JS} =\left(1-\frac{n-2}{\|\mathbf{x}-\bar x\|^2}\right)x_i,
$$ 

where 

$$
\|x-\bar x\|^2=\sum_{i=1}^{n}(x_i-\bar x)^2.
$$


The James Stein estimator can also be written as $$ \hat \mu_i^\text{JS} =\bar x + \lambda^+(x_i-\bar x) $$ with $$\lambda^+=(1-\frac{n-3}{\|\mathbf{x}-\bar x\|^2})^{+}$$. As defined, $$\lambda^+\in[0,1]$$, you can already see that, the James Stein estimator shrinks the first manoeuvre performance of a flight cadet by a factor $$\lambda^{+}$$ towards the mean $$\bar x$$.  The James Stein estimator clearly acknowledges the existence of the regression to the mean, in its construct.

As we will expect, the overall mean squared error in estimating the actual manoeuvre performance of a flight cadet with the James Stein is smaller than that of the MLE. The James Stein estimator suggests poorer true manoeuvre performances for outstanding observed manoeuvre performances and better true manoeuvre performances for abysmal manoeuvre performances.

## Bayesian view of the James Stein estimator
Bradley Efron in several of his papers and books has garnished the James Stein estimator with Bayesian flavour. Bradley Efron is Max H. Stein Professor of Humanities and Sciences at Stanford University and the inventor of the famous \textit{bootstrap} technique.

Now consider that the true manoeuvre performances of the cadets is governed as follows $$\mu_i\sim N(0,A)$$, $$A>$$ (in Bayesian terminology -- the prior). And, as we already know, we assume $$x_i\mid \mu_i\sim N(\mu_i,1)$$. 
The marginal distribution of $$x_i$$ obtained from integrating $$x_i\mid \mu_i\sim N(\mu_i,1)$$ over $$\mu_i\sim N(0,A)$$ is 

$$
x_i\sim N(0,A+1).
$$

This leads to the Bayes estimator,

$$
E(\mu_i\mid x_i)=x_i-\frac{x_i}{A+1}=\left(1-\frac{1}{A+1}\right)x_i.
$$

Again, we realise that the Bayes estimator of $$\mu_i$$,  $$E(\mu_i\mid x_i)$$, shrinks the $$x_i$$ towards zero. If $$A=1$$, the $$x_i$$ is shrunk by half. On the other hand, if $$A=0$$, the Bayes estimator is zero almost surely. If we let

$$
\frac{1}{A+1}=\frac{n-2}{\|x-\bar x\|^2},
$$

then $$E(\mu_i\mid x_i)$$ becomes the James Stein estimator. With some straightforward algebra, $$(n-2)/\|x-\bar x\|^2$$  is an unbiased estimator of $$\frac{1}{A+1}$$. 

Fisherians, i.e.,  defenders of the MLE  might then say that of course, we expect the James Stein estimator to perform better than the MLE. Why? Because if you look at the James Stein estimator through a Bayesian lens as Efron did, the James Stein estimator makes the additional assumption of $$\mu_i\sim N(0, A)$$. The MLE does not make this assumption and we, therefore, expect the James Stein estimator to perform (smaller mean squared errors)  better when this assumption is not violated. James Stein in their famous theorem showed that even without the assumption $$\mu_i\sim N(0,A)$$, the James Stein estimator is still a better estimator than the MLE. The theorem states.

## James-Stein Theorem

Suppose $$x_i\mid \mu_i\sim N(\mu_i,1)$$ independently for $$i=1,\ldots,n$$, $$n > 3$$. Then 

$$
E(\|\hat \mu^\text{JS}-\mu\|^2)-E(\|\hat \mu^\text{MLE}-\mu\|^2).
$$

Where $$\mu=(\mu_1,\ldots,\mu_n)$$,  $$\hat \mu^\text{JS}=(\hat \mu_1^\text{JS},\ldots,\hat \mu_n^\text{JS})$$, and $$\hat \mu^\text{MLE}=(\hat \mu_1^\text{MLE},\ldots,\hat \mu_n^\text{MLE})$$.

Using the terminology of decision theory, this theorem says the MLE is inadmissible. That is, the risk of the MLE is not the least, the risk of the James Stein estimator as the theorem states is smaller. This theorem does not make any prior distribution assumptions about $$\mu_i$$ and hence no Fisherian can critisize the James Stein estimator.

We will conclude this discussion by saying that because of regression to the mean, the MLE is inadmissible. The James Stein estimator is a better estimator since it takes regression to the mean into account. 
