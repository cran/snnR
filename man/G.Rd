\name{G}
\alias{G}
\title{Genomic additive relationship matrix for the Jersey dataset.}
\description{

A matrix has been used in Gianola et al. (2011) for predicting 
milk, fat and protein production in Jersey cows. In this software version we do not center the incidence 
matrix for the additive effects.

\deqn{G=\frac{X_a X_a'}{2\sum_{j=1}^p p_j (1-p_j)},}

where

 \itemize{
     \item{\eqn{X_a} is the design matrix for allele substitution effects for additivity.}
     \item{\eqn{p_j} is the frecuency of the second allele at locus \eqn{j} and \eqn{q_j=1-p_j}.}
  }

}

\source{
  University of Wisconsin at Madison, USA.
}

\references{
Gianola, D. Okut, H., Weigel, K. and Rosa, G. 2011. 
"Predicting complet quantitative traits with Bayesian neural networks:
a case study with Jersey cows and wheat". \emph{BMC Genetics} 12(1), 87-92.
.
}

