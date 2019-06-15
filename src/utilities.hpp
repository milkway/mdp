#ifndef __UTILITIES__
#define __UTILITIES__

// C++ Libraries
#include <omp.h>
#include <sstream>
#include <chrono>

/********************
 * Constants
 ********************/
// Alpha multiplier for tabu tenure list
static const arma::uvec alpha_multiplier {1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1};


/*********************
 * Functions Headers *
 *********************/
// Execute a Tabu Search in the neighborhood of Solution S (MaxIter Criterium)
// Algorithm 2. Constrained neighborhood tabu search procedure for MDP 
// Ref: http://dx.doi.org/10.1016/j.ejor.2013.06.002
Rcpp::List cnts(arma::uvec S, double fitness, const arma::mat& distances, unsigned alpha, double rho, unsigned max_iterations, bool verbose);
Rcpp::List cnts_sugar(arma::uvec S, const arma::mat& distances, unsigned alpha, double rho, unsigned max_iterations, bool verbose);
// Tour fitness from binary representation
double tour_fitness_binary(const arma::uvec& S, const arma::mat& distances);
// Algorithm 3. The crossover operator of the MAMDP algorithm
// Ref: http://dx.doi.org/10.1016/j.ejor.2013.06.002
arma::uvec crossover(const arma::uvec& S_a, const arma::uvec& S_b, const arma::mat& distances); 
arma::uvec crossover_rand(const arma::uvec& S_a, const arma::uvec& S_b, const arma::mat& distances); 
// Eq. 8 of http://dx.doi.org/10.1016/j.ejor.2013.06.002
// Scale Function used in population update
arma::vec normalized_function(arma::vec input);
// Def. 1  of http://dx.doi.org/10.1016/j.ejor.2013.06.002
// Distance between to individuals;
double solutions_distance(const arma::uvec& S0, const arma::uvec& S1);
// Def. 2  of http://dx.doi.org/10.1016/j.ejor.2013.06.002
// Distance between individual and a population;
double solution_to_population_distance(const arma::uvec& S0, const arma::umat& population, unsigned tour_size);
// Def. 2  of http://dx.doi.org/10.1016/j.ejor.2013.06.002
// Distance between individual and a population;
double index_to_population_distance(const unsigned idx, const arma::umat& population);
// Tabu search in a random solution
Rcpp::List tabu_solution_rand(const arma::mat& distances, unsigned tour_size, unsigned alpha, double rho, unsigned max_iterations);
// Algorithm 4. Update population strategy of the MAMDP algorithm
// Ref: http://dx.doi.org/10.1016/j.ejor.2013.06.002
Rcpp::List update_population_mamdp(arma::uvec S, arma::umat& population, arma::vec& fitness, const arma::mat& distances, double beta);   
// Pool initialization for MAMDP
// Section 2.2 of http://dx.doi.org/10.1016/j.ejor.2013.06.002
Rcpp::List initialize_population_mamdp(const arma::mat& distances, unsigned tour_size, unsigned population_size, unsigned tabu_max_iterations, unsigned tabu_multiplier, double tabu_rho, unsigned tabu_alpha);   


#endif //__UTILITIES__