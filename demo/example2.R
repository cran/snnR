  #Jersey dataset
  data(Jersey) 
  #Fit the model with additive effects
  y<-as.vector(pheno$yield_devMilk)
  X_test<-G[partitions==2,]
  X_train<-G[partitions!=2,]
  y_test<-y[partitions==2]
  y_train<-y[partitions!=2]
  #Generate the structure of neural network   
  nHidden <- matrix(c(5,5,5),1,3)
  #call function to train the sparse nerual network 
  network=snnR(x=X_train,y=y_train,nHidden=nHidden,iteramax =10,normalize=TRUE)
  #predictive results
  yhat= predict(network,X_test)
  plot(y_test,yhat)



