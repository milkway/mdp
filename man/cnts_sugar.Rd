% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{cnts_sugar}
\alias{cnts_sugar}
\title{Execute a Tabu Search in the neighborhood of Solution S (Max Iterations)}
\usage{
cnts_sugar(S, distances, alpha = 15L, rho = 1,
  max_iterations = 1000L, verbose = FALSE)
}
\arguments{
\item{\code{S}}{Initial solution.}

\item{\code{fitness}}{Initial solution fitness variable.}

\item{\code{distances}}{Square and symmetric distance matrix.}

\item{\code{alpha}}{Tenure list multiplier.}

\item{\code{rho}}{Neighborhood constraint coefficient.}

\item{\code{max_iteration}}{number of search iterations.}

\item{\code{verbose}}{print results}
}
\description{
Execute a Tabu Search in the neighborhood of Solution S (Max Iterations)
}
\examples{
cnts()
}
