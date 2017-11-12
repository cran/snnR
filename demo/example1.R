###############################################################
#Example 1 
  #Nonlinear function  regression
  library(snnR)
  #Generate the data
  nsamples<-200
  nvaibles<-1
  Xydata<-SimData("Nonlinearregress",nsamples,nvaibles)
  x<-as.matrix(Xydata$X) 
  y<-as.vector(Xydata$y)
  #Generate the structure of neural network
  #5 hidden layers and 5 or 15 neurons in each layer
  nHidden <- matrix(c(5,5,5),1,3)
  # call function to train the sparse nerual network
  network=snnR(x=x,y=y,nHidden=nHidden,iteramax =10)
  # test data
  X_test<-as.matrix(seq(-5,5,0.05))
  #  predictive results
  yhat=predict(network,X_test)
  split.screen(c(1,2))
  screen(1)
  plot(x,y)
  screen(2)
  plot(X_test,yhat)
  ### please install R package NeuralNetTools to show the optimal structure of NN
  ### and use the following commands
  #library(NeuralNetTools)
  #optstru=write.NeuralNetTools(w =network$wDNNs,nHidden =nHidden ,x = x,y=y )
  #plotnet(optstru$w_re,struct = optstru$structure)

