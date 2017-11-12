\name{write.NeuralNetTools}
\alias{write.NeuralNetTools}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Prepare data for NeuralNetTools
}
\description{
Create input files and corresponding script  for NeuralNetTools soft-
ware (An R package at the CRAN site) to show the  structure of NNs
using parameters (weights and biases).
}
\usage{
write.NeuralNetTools(w,nHidden,x,y)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{w}{ (\code{numeric}) parameters (weights and biases) of neural network.}
  \item{nHidden}{(positive integer, \eqn{1\times h}{1 x h}) matrix, h indicates the  number of hidden-layers and nHidden[1,h] indicates the  neurons of the h-th hidden-layer.}
  \item{x}{((numeric, \eqn{n \times p}{n x p}) incidence matrix.}
  \item{y}{(numeric, \eqn{n}{n}) the response data-vector.}
}
\value{
 Input data for NeuralNetTools.
}
\references{
Beck MW. 2015. NeuralNetTools: Visualization and Analysis Tools for Neural Networks. Version 1.5.0. https://cran.rstudio.com/package=NeuralNetTools
}

\examples{
\dontrun{
  library(NeuralNetTools)
  optstru=write.NeuralNetTools(w =network$wDNNs,nHidden =nHidden ,x = x,y=y )
  plotnet(optstru$w_re,struct = optstru$structure)
}
}


