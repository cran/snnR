\name{predict.snnR}
\alias{predict.snnR}
\title{predict.snnR}
\description{
  The function produces the predictions for a n-layer feed-forward neural network.
}
\usage{
   \method{predict}{snnR}(object,newdata,\dots)
}
\arguments{
  \item{object}{an object of the class \code{snnR} as returned by \code{snnR}}
  \item{newdata}{matrix  of test examples. }
  \item{\dots}{arguments passed to or from other methods.}
 }

\details{
  This function is a method for the generic function
  \code{predict()} for class \code{"snnR"}.
	It can be invoked by calling \code{predict(x)} for an
	object \code{x} of the appropriate class, or directly by
	calling \code{predict.snnR(x)} regardless of the class of the object.
}

\value{
  A vector containing the predictions
}

\keyword{models}
