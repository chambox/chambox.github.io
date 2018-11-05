p<-10
N<-100

beta<-c(c(-3,-2),rep(0,6),c(3,2))

Rep=1000
Err_App<-rep(NA,Rep)
Err_cv<-rep(NA,Rep)
Err_val<-rep(NA,Rep)
for(ss in 1:Rep)
{
  X<-matrix(rep(NA,p*N),ncol = p)
  X_val<-matrix(rep(NA,p*N),ncol = p)
  for(i in 1:p)
  {
    X[,i]<-runif(N,0,1)
    X_val[,i]<-runif(N,0,1)
  }
  y<-10+X%*%beta+rnorm(N,0,1)
  y_val<-10+X_val%*%beta+rnorm(N,0,1)
  
  fitcv1<-cv.glmnet(X,y,nfolds = 10,type.measure = "mse") 
  Err_cv[ss]=fitcv1$cvm[fitcv1$lambda==fitcv1$lambda.1se]
  
  yhat_val<-predict(fitcv1,newx= X_val)
  yhat<-predict(fitcv1,newx=X)
  Err_App[ss]<-mean((y-yhat)^2)
  Err_val[ss]<-mean((yhat_val-y_val)^2)
}

plot(Err_cv,Err_val)
apply(cbind(Err_cv,Err_val,Err_App),2,mean)
apply(cbind(Err_cv,Err_val,Err_App),2,sd)
cor(Err_cv,Err_val)
plot(Err_val,Err_App)
