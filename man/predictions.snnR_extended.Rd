\name{predict.snnR_extended}
\alias{predict.snnR_extended}
\title{predict.snnR_extended}
\description{
  The function produces the predictions for a n-layer feed-forward neural network with  additive and dominance effects.
}
\usage{
   \method{predict}{snnR_extended}(object,newdata_add,newdata_dom,\dots)
}
\arguments{
  \item{object}{an object of the class \code{snnR_extended} as returned by \code{snnR_extended}}
  \item{newdata_add}{matrix  of test examples for additive effects.}
  \item{newdata_dom}{matrix  of test examples for dominance effects.}
  \item{\dots}{arguments passed to or from other methods.}
 }

\details{
  This function is a method for the generic function
  \code{predict()} for class \code{"snnR_extended"}.
	It can be invoked by calling \code{predict(x)} for an
	object \code{x} of the appropriate class, or directly by
	calling \code{predict.snnR_extended(x)} regardless of the class of the object.
}

\value{
  A vector containig the predictions
}

\keyword{models}
