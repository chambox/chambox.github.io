---
layout: post
title: Validating prediction models
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>



Nowadays machine learning (ML) and artificial intelligence (AI) are integral to most businesses. Decision-makers in many sectors, e.g., banking and finance, have employed ML-algorithms to do their heavy lifting. Though it sounds like a smart move, it is imperative to make sure these models are indeed doing what is expected of them. As employees of a bank, models are prone to mistakes and there is always a price to pay when we use ML-models. Hence, there is a need to validate models.

Model validation is a broad field and may cover several notions. In this article, we narrow down to prediction models. A model may be considered valid if:


1. it performs well, i.e., based on some mathematical metrics such as “a small miss-classification error” in classification prediction models.
2. the model is fair, i.e., not racist, sexist, homophobes, xenophobe, etc. These are cultural or legal validity aspects
3. the model is interpret-able, some experts may argue that black-box models are invalid. In such cases understanding how the model makes predictions is central


The three validity aspects above are not exhaustive; model validation may mean different things depending on who is the validator.

In this article, we will discuss model validation from the viewpoint of 1. Most data scientist when talking about model validation will default to point 1. Hereunder, we give models details on model validation based on prediction errors.


## Validating prediction models based on errors in prediction

Before making any progress, we will introduce some notations here:

$$Y$$: will represent the outcome we want to predict, let's say something like stock prices on a given day. We will denote the predicted $$Y$$ with $$\hat Y $$

$$x:$$  will represent the characteristics of the outcome — we will always know $$x$$ at the time prediction.  For our stock example,
$$x$$ can be a date, open and closing prices for that date and so on.

$$m$$:  will represent a prediction model, in practice, this model will often contain parameters which is our job to estimate them. Once we estimate these parameters, we then denote the model with estimated parameters with $$\hat m$$ — this will differentiate $$\hat m$$ from $$m$$, the model with the true parameters.

$$\beta:$$  will represent parameters of the model $$m$$, similarly $$\hat \beta$$ will represent the parameters of $$\hat m$$. 

We calculate predictions as follows:


$$
\hat Y(x) = \hat m (x) = x^t \hat \beta
$$


and want the prediction error to be as small as possible. The prediction error for a prediction at predictor $$x$$ is given by

$$
\hat Y(x)-Y^{\star}
$$

$$Y^\star$$ is the outcome we want to predict that has $$x$$ as characteristics. Since a prediction model is typically used to predict an outcome before it is observed, the outcome $$Y^\star$$ is unknown at the time of prediction. Hence, the prediction error cannot be computed.



*Recall that a prediction model is estimated by using data from the training data set $$(X, Y)$$ and that $$Y^\star$$ is an outcome at $$x$$ which is assumed to be independent of the training data. The idea is that the prediction model is intended for use in predicting a future observation $$Y^\star$$, i.e., an observation that still has to be realized/observed (otherwise prediction seems rather useless). Hence, $$Y^\star$$  can never be part of the training data set.*

## Evaluating prediction performance

Here we provide definitions and we show how the prediction performance of a prediction model can be evaluated from data.

Let $$T= (Y; X)$$ denote the training data, from which the prediction model is built. The building process typically involves feature (characteristic) selection and parameter estimation. Below we define different types of errors used for model validation.



### Test or Generalisation, outsample Error

The test or generalization error for prediction model is given by

$$
\text{Err}_T = \text{E}_{Y^\star,X^\star}\big\{ (\hat m(X^\star)-Y^\star)^2|T\big\}
$$

where $$(Y^\star, X^\star)$$ is independent of the training data.

The test error is conditional on the training data $$T$$. Hence, the test error evaluates the performance of the single model built from the observed training data. This is the ultimate target of the model assessment because it is exactly this prediction model that will be used in practice and applied to future predictors $$X^\star$$ to predict $$Y^\star$$. The test error is defined as an average over all such future observations $$(Y^\star, X^\star)$$. The test error is the most interesting error for model validation according to point .1 above.

### Conditional Test Error in $$x$$ 


The conditional test error in $$x$$ for a prediction model $$\hat m$$ is given by


$$
\text{Err}_T(x) = \text{E}_{Y^\star}\big\{ (\hat m(x)-Y^\star)^2|T,x\big\}
$$

where $$Y^\star$$ is an outcome at predictor $$x$$, independent of the training data.

### In-simple Error

The in-sample error for a prediction model is given by


$$
\text{Err}_{\text{in}T} = \frac{1}{n}\sum_{i=1}^n \text{Err}_{T}(x_i),
$$



i.e., the in-sample error is the sample average of the conditional test errors evaluated in the n training dataset observations.

### Estimation of the in-sample error

We start with introducing the training error rate, which is closely related to the mean squared error in linear models

#### Training error
The training error is given by
$$
\bar {\text{err}} = \frac{1}{n}\sum_{i=1}^n(Y_i-\hat m (x_i))^2,
$$


where the $$(Y, x)$$ form the training dataset which is also used for training the models.
* The training error is an overly optimistic estimate of the test error
* The training error never increase when the model becomes more complex — cannot be used directly as a model selection criterion


*Model parameters are often estimated by minimising the training error (cfr. mean squared error). Hence the fitted model adapts to the training data, and hence the training error will be an overly optimistic estimate of the test error*


Other estimators of the in-sample error are:

* The Akaike information criterion (AIC)
* Bayesian information criterion (BIC)
* Mallow’s Cp

## Expected Prediction Errors and Cross-Validation

The test or generalization error was defined condition on the training data. By averaging over the distribution of training datasets, the expected test error arises.

### Expected Test Error

$$
\text{E}_T\text{Err}_T = \text{E}_T\big\{\text{E}_{Y^\star,X^\star}\big\{ (\hat m(X^\star)-Y^\star)^2|T\big\}\}
$$
$$
= \text{E}_{Y^\star,X^\star,T}\big\{ (\hat m(X^\star)-Y^\star)^2\}
$$

*The expected test error may not be of direct interest when the goal is to assess the prediction performance of a single prediction model. The expected test error averages the test errors of all models that can be build from all training datasets, and hence this may be less relevant when the interest is in evaluating one particular model that resulted from a single observed training dataset.*


Also, note that building a prediction model involves both parameter estimation and feature selection. Hence the expected test error also evaluates the feature selection procedure (on average). If the expected test error is small, it is an indication that the model building
process gives good predictions for future observations $$(Y^\star; X^\star)$$ on average.


#### Estimation of the expected test error

*Cross-validation*

Randomly divide the training dataset into $$k$$ approximately equal subsets. Train your model on $$k-1$$ of them and compute the prediction error on the $$k$$th subset. Do this in such a way that all $$k$$ subsets have been used for training and for computing the prediction error. Average the prediction errors from all $$k$$ subset. This is the cross-validation error.



*The process of cross-validation is clearly mimicking an expected test error and not the test error. Clearly, cross-validation is estimating the expected test error and not the test error in which we are interested in.*


## Conclusion

Validating prediction models based on prediction errors can get complicated very quickly due to the different types of prediction errors that exist. A data scientist that uses any of these errors during model validation should be conscious of it. Ideally, the model validation error of interest is the **test error**. Proxies like the in-sample and generalized test error are often used in practice — when used their shortcomings should be clearly stated and the reason for using them should be also outlined.
If there is enough data, you could, in fact, partition your data into training, validation( for hyper-parameter tuning) and perhaps multiple testing sets. This way good estimates of the test error are possible, leading to proper model validation.

