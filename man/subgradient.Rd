\name{subgradient}
\alias{subgradient}
\title{subgradient}
\description{
This function obtains the minimum-norm subgradient of the approximated square error with L1 norm penalty or L2 norm penalty.
}
\usage{subgradient(w, X, y, nHidden, lambda, lambda2)}
\arguments{
  \item{w}{(numeric, \eqn{n}{n}) weights and biases.}
  \item{X}{(numeric, \eqn{n \times p}{n x p}) incidence matrix.}
  \item{y}{(numeric, \eqn{n}{n}) the response data-vector.}
  \item{nHidden}{(positive integer, \eqn{1\times h}{1 x h}) matrix, h indicates the  number of hidden-layers and nHidden[1,h] indicates the 
 neurons of the h-th hidden-layer.}
  \item{lambda}{(numeric,\eqn{n}{n}) lagrange multiplier for L1 norm penalty on parameters.}
  \item{lambda2}{(numeric,\eqn{n}{n}) lagrange multiplier for L2 norm penalty on parameters.}
}
\details{
It is based on choosing a subgradient with minimum norm  as a steepest descent direction and taking a step resembling Newton iteration in this direction with a Hessian approximation.
}
\value{
 A vector with the subgradient values.
}

