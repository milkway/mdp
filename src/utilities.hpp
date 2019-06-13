#ifndef __UTILITIES__
#define __UTILITIES__

/********************
 * Constants
 ********************/
// Alpha multiplier for tabu tenure list
static const arma::uvec alpha_multiplier {1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1};


/*********************
 * Functions Headers *
 *********************/
// Execute a Tabu Search in the neighborhood of Solution S (MaxIter Criterium)
// Algorithm 2 of http://dx.doi.org/10.1016/j.ejor.2013.06.002
Rcpp::List cnts(arma::uvec S, double fitness, const arma::mat& distances, int alpha, double rho, int max_iterations, bool verbose);
Rcpp::List cnts_sugar(arma::uvec S, const arma::mat& distances, int alpha, double rho, int max_iterations, bool verbose);
double tour_fitness_binary(const arma::uvec& S, const arma::mat& distances);

#endif //__UTILITIES__