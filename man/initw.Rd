\name{initw}
\alias{initw}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Initialize weights and biases of a neural network. 
%%  ~~function to do ... ~~
}
\description{
Function to initialize the weights and biases in a neural network. 
}
\usage{
initw(nHidden, nvaibles)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{nHidden}{(positive integer, \eqn{1\times h}{1 x h}) matrix that indicates number of neurons and number layers of neural network.
%%     ~~Describe \code{nHidden} here~~
}
  \item{nvaibles}{(positive integer) number of parameters to be estimate including only weights and biases.
%%     ~~Describe \code{nvaibles} here~~
}
}

\value{
 A list containing initial values for weights and biases.
}

