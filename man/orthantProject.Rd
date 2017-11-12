\name{orthantProject}
\alias{orthantProject}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{orthantProject
%%  ~~function to do ... ~~
}
\description{This function  obtains  an orthant projection.
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
orthantProject(w, xi)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{w}{(numeric,\eqn{n}{n}) weights and biases data-vector.
%%     ~~Describe \code{w} here~~
}
  \item{xi}{(numeric,\eqn{n}{n}) weights and biases data-vector.
%%     ~~Describe \code{xi} here~~
}
}
\details{

  \eqn{  P_{O}(X+h)= \{ 0 \quad if \quad X_i(X_i+h_i)<0; \quad  X_i+h_i  \quad if \quad otherwise\}}

}
\value{
A vector of weights and biases containing zero values.
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}

