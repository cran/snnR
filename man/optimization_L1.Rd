\name{optimization_L1}
\alias{optimization_L1}
\title{optimization_L1
%%  ~~function to do ... ~~
}
\description{Subgradient-based quasi-Newton method for non-differentiable optimization.

}
\usage{
optimization_L1(w,X,y,nHidden, verbose= FALSE,lambda,lambda2, optimtol, prgmtol,
                      maxIter, decsuff)
}

\arguments{
  \item{w}{(numeric,\eqn{n}{n}) weights and biases.}
  \item{X}{(numeric, \eqn{n \times p}{n x p}) incidence matrix.}
  \item{y}{(numeric, \eqn{n}{n}) the response data-vector.}
  \item{nHidden}{(positive integer, \eqn{1\times h}{1 x h}) matrix, h indicates the  number of hidden-layers and nHidden[1,h] indicates the 
 neurons of the hth hidden-layer.}
  \item{verbose}{logical, if TRUE prints detail history.}
  \item{lambda}{numeric, lagrange multiplier for L1 norm penalty on parameters.}
  \item{lambda2}{numeric, lagrange multiplier for L2 norm penalty on parameters.}
  \item{optimtol}{numeric, a tiny number useful for checking convergenge of subgradients.}
  \item{prgmtol}{numeric, a tiny number useful for checking convergenge of parameters of NN.}
  \item{maxIter}{positive integer, maximum number of epochs(iterations) to train, default 100.}
  \item{decsuff}{numeric, a tiny number useful for checking change of loss function.}
}
\details{
It is based on choosing a sub-gradient with minimum norm as a steepest descent direction and taking a step resembling Newton iteration in this direction with a Hessian approximation (Nocedal, 1980). An active-set method is adopted to set some parameters to exactly zero (Krishnan et al., 2007). At each iteration, the non-zero parameters are divided into two sets: the working set containing the non-zero variables, and the active set containing the sufficiently zero-values variables. Then a Newton step is taken along the working set. A subgradient-based quasi-Newton method ensures that the step size taken in the active variables is such that they do not violate the sufficiently zero-value variables constraint.  A projected steepest descent is taken to set some parameters to exactly zero.
}

\references{
Nocedal, J. 1980. Updating quasi-newton matrices with limited storage. \emph{Mathematics of Computation}, 35(35), 773-782.

Krishnan, D., Lin, P., and Yip,  A., M. 2007. A primal-dual active-set method for non-negativity constrained total variation deblurring problems. \emph{IEEE Transactions on Image Processing}, 16(11), 2766-2777.

}


\value{A vector of weights and biases.
}

