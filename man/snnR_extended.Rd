\name{snnR_extended}
\alias{snnR_extended}
\title{snnR_extended}
\description{
  The snnR_extended function fits a sparse neural network for a GS model with additive and dominance efforts. It uses the sub-gradient and active-set methods to perform the optimization.
}
\usage{
snnR_extended(x,y,z,nHidden_add,nHidden_dom,normalize=TRUE,verbose=TRUE, optimtol = 1e-5, 
 prgmtol = 1e-9, iteramax = 20, decsuff =1e-4,lambda)
}
\arguments{
  \item{x}{(numeric, \eqn{n \times p}{n x p}) incidence matrix for additive effects.}
  \item{y}{(numeric, \eqn{n}{n}) the response data-vector (NAs not  allowed).}
  \item{z}{(numeric, \eqn{n \times p}{n x p}) incidence matrix for dominance effects.}
  \item{nHidden_add}{(positive integer, \eqn{1\times h}{1 x h}) matrix, h indicates the  number of hidden-layers and nHidden[1,h] indicates the neurons of the hth hidden-layer.}
  \item{nHidden_dom}{(positive integer, \eqn{1\times h}{1 x h}) matrix, h indicates the  number of hidden-layers and nHidden[1,h] indicates the neurons of the hth hidden-layer.}
  \item{normalize}{logical, if TRUE normalizes output, the default value is FALSE. }
  \item{verbose}{logical, if TRUE prints detail history.}
  \item{optimtol}{numeric, a tiny number useful for checking convergenge of subgradients.}
  \item{prgmtol}{numeric, a tiny number useful for checking convergenge of parameters of NN.}
  \item{iteramax}{positive integer, maximum number of epochs(iterations) to train, default 20.}
  \item{decsuff}{numeric, a tiny number useful for checking change of loss function.}
  \item{lambda}{numeric, L1 norm lagrange multiplier.}
}

\value{
     Mostly internal structure, but it is a list containing:

\item{$wDNNs_add}{A list containing weights and biases for additive effects.}
\item{$wDNNs_dom}{A list containing weights and biases for dominance effects.}
\item{$inputwgts_add}{A list containing input weights and biases for additive effects.}
\item{$outputwgts_add}{A list containing output weights and biases for additive effects.}
\item{$hidewgts_add}{A list containing hidden weights and biases for additive effects.}
\item{$inputwgts_dom}{A list containing input weights and biases for dominance effects.}
\item{$outputwgts_dom}{A list containing output weights and biases for dominance effects.}
\item{$hidewgts_dom}{A list containing hidden weights and biases for dominance effects.}
\item{$Mse=Mse}{The mean squared error between observed and predicted values.}
\item{$message}{String that indicates the stopping criteria for the training process.}
}

\examples{
\donttest{
###############################################################
#Example 1
  #Jersey dataset
   data(Jersey) 
  #Fit the model with Additive and Dominant effects
   y<-as.vector(pheno$yield_devMilk)
   X_test<-G[partitions==2,]
   X_train<-G[partitions!=2,]
   y_test<-y[partitions==2]
   y_train<-y[partitions!=2]
   Z_test<-D[partitions==2,]
   Z_train<-D[partitions!=2,]
   #Generate the structure of neural network   
   nHidden_add <- matrix(c(5,10,5),1,3)
   nHidden_dom <- matrix(c(5,15,5),1,3)
   # call function to train the sparse nerual network 
   network=snnR_extended(x=X_train,y=y_train,
   z=Z_train,nHidden_add=nHidden_add,nHidden_dom=nHidden_dom,iteramax =10,normalize=TRUE)
   # predictive results
   yhat= predict(network,X_test,Z_test)
   plot(y_test,yhat)

}

}

