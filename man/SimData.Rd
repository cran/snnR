\name{SimData}
\alias{SimData}
\title{SimData}
\description{
 Function to create simulation data of nonlinear function or nonlinear classification.
}
\usage{
   SimData(type,nsamples,nvaibles,nClasses) 
}
\arguments{
  \item{type}{If type="Nonlinearregress", this function  creates simulation data of nonlinear function. If type="Nonlinearregress", this function creates simulation data of nonlinear classification.}
  \item{nsamples}{positive integer, the samples size of data.}
 \item{nvaibles}{positive integer, the vaible size of data.}
  \item{nClasses}{positive integer, the number of classes for nonlinear classification.}
}

\examples{
\dontrun{
###############################################################
#Example 1
 nsamples = 200
 nvaibles = 1
 # Simulation data of nonlinear function with 200 samples and 1 varible input.
 NonlinregressData=SimData("Nonlinearregress",nsamples,nvaibles)
 #  Simulation data of nonlinear classification with 200 samples and 1 varible input for 2 classes.
 nclass=2
 ClassificationData=SimData("Nonlinearclassification",nsamples,nvaibles,nclass)
}
}
\value{
  A vector of simulation data of nonlinear function or nonlinear classification .
}


