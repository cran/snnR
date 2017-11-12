\name{snnR}
\alias{snnR}
\title{snnR}
\description{
 The snnR function fits a sparse neural network by minimizing the square error subject to a penalty on the L1-norm of the parameters (weights and biases) to solve the over-parameterization in NN for improving the predictive performance. It is based on choosing a subgradient
with minimum norm as a steepest descent and using  an active-set method  to set many of the parameters to exactly zero.


}
\usage{
 snnR(x, y, nHidden, normalize=FALSE, verbose=TRUE, optimtol = 1e-5, prgmtol = 1e-9, 
 iteramax = 100, decsuff =1e-4,lambda)
}
\arguments{
  \item{x}{(numeric, \eqn{n \times p}{n x p}) incidence matrix.}
  \item{y}{(numeric, \eqn{n}{n}) the response data-vector (NAs not  allowed).}
  \item{nHidden}{(positive integer, \eqn{1\times h}{1 x h}) matrix, h indicates the  number of hidden-layers and nHidden[1,h] indicates the 
 neurons of the h-th hidden-layer.}
  \item{normalize}{logical, if TRUE normalizes output, the default value is FALSE.}
  \item{verbose}{logical, if TRUE prints detail history.}
  \item{optimtol}{numeric, a tiny number useful for checking convergenge of subgradients.}
  \item{prgmtol}{numeric, a tiny number useful for checking convergenge of parameters of NN.}
  \item{iteramax}{positive integer, maximum number of epochs(iterations) to train, default 100.}
  \item{decsuff}{numeric, a tiny number useful for checking change of loss function.}
  \item{lambda}{numeric, L1 norm lagrange multiplier.}
}
\details{
 The software fits sparse deep neural networks including  a two layer network as described in Gianola (2011).
  The two layer network model is given by:
  
  \eqn{y_i=g(\boldsymbol{x}_i)+e_i = \sum_{k=1}^s w_k g_k (b_k + \sum_{j=1}^p x_{ij} \beta_j^{[k]}) + e_i, i=1,...,n}
  
  where:
  \itemize{
    \item{\eqn{e_i \sim N(0,\sigma_e^2)}{e_i ~ N(0,\sigma_e^2)}.}
    \item{\eqn{s} is the number of neurons.}
    \item{\eqn{w_k}{w_k} is the weight of the \eqn{k}{k}-th neuron, \eqn{k=1,...,s}{k=1,...,s}.}
    \item{\eqn{b_k}{b_k} is a bias for the \eqn{k}{k}-th neuron, \eqn{k=1,...,s}{k=1,...,s}.}
    \item{\eqn{\beta_j^{[k]}}{beta_j^{[s]}} is the weight of the \eqn{j}{j}-th input to the net, \eqn{j=1,...,p}{j=1,...,p}.}
    \item{\eqn{g_k(\cdot)}{g_k(.)} is the activation function, in this implementation \eqn{g_k(x)=\frac{\exp(2x)-1}{\exp(2x)+1}}{g_k(x)=(exp(2x)-1)/(exp(2x)+1)}.}
   }  
   For estimating sparse SLNN,  the parameters are typically trained by minimizing the approximated least squares error subject to the sum of the absolute parameters being less than some constants:
   \deqn{\min\limits_{\bm{W},\bm{b},\bm{\beta}} \hat{\bm{L}}(\bm{W}, \bm{b}, \bm{\beta}) = \sum\limits_{i=1}^{n}\bigg{(}\mu+\bm{W}^T \bm{g}\big{(}\bm{b}+\bm{\beta}\bm{x_i}\big{)}-y_i\bigg{)}^2}
   \deqn{subject \quad to \quad  \sum\limits_{j=1}^{\mathcal{S}}\sum\limits_{k=1}^{p} |\beta_{j}^{[k]}|\leq t_1, \sum\limits_{k=1}^{\mathcal{S}}|W_k|\leq t_2,
  \sum\limits_{k=1}^{\mathcal{S}}|b_k|\leq t_3.}
  
}
\value{
    
object of class \code{"snnR"}. Mostly internal structure, but it is a list containing:
\item{$wDNNs}{A list containing weights and biases.}
\item{$inputwgts}{A list containing input weights and biases.}
\item{$outputwgts}{A list containing output weights and biases.}
\item{$hidewgts}{A list containing hidden weights and biases.}
\item{$Mse=Mse}{The mean squared error between observed and predicted values.}
\item{$message}{String that indicates the stopping criteria for the training process.}
}
\references{
Gianola, D. Okut, H., Weigel, K. and Rosa, G. 2011. "Predicting complex quantitative traits with Bayesian neural networks:a case study with Jersey cows and wheat". \emph{BMC Genetics}, 12(1), 87-92.


Perez-Rodriguez, P., Gianola, D., Weigel, K. A., Rosa, G. J., and  Crossa, J. 2013. Technical note: an r package for fitting bayesian regularized neural networks with applications in animal breeding. \emph{Journal of Animal Science}, 91(8), 3522-3531.


Krishnan, D., Lin, P., and Yip,  A., M. 2007. A primal-dual active-set method for non-negativity constrained total variation deblurring problems. \emph{IEEE Transactions on Image Processing}, 16(11), 2766-2777.


Nocedal, J. 1980. Updating quasi-newton matrices with limited storage. \emph{Mathematics of Computation}, 35(35), 773-782.

Wang, Y., Lin, P., and Wang, L. 2012. Exponential stability of reaction-diffusion high-order markovian jump hopfield neural networks with time-varying delays. \emph{Nonlinear Analysis B}, 13(3), 1353-1361.


Liang, X., Wang, L., Wang, Y., and Wang, R. 2015. Dynamical behavior of delayed reaction-diffusion hopfield neural networks driven by infinite dimensional wiener processes. \emph{IEEE Transactions on Neural Networks}, 27(9), 1816-1826.

}

\examples{
 \dontshow{
   #examples for checks:
   #executable in < 5 sec together with the examples above not shown to users
   #examples for users:
   #Nonlinear function  regression
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
  nHidden <- matrix(c(5,5),1,2)
  # call function to train the sparse nerual network
  network=snnR(x=x,y=y,nHidden=nHidden,iteramax =8)
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
   }
\donttest{
#Load the library
library(snnR)
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
  nHidden <- matrix(c(5,5,15,5,5),1,5)
  # call function to train the sparse nerual network
  network=snnR(x=x,y=y,nHidden=nHidden)
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

###############################################################
#Example 2
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


}

}

