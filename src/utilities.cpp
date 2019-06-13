/*
 * André Leite <leite@de.ufpe.br>
 * Geiza Silva <geiza@de.ufpe.br>
 */

#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

// C++ Libraries
#include <omp.h>
#include <sstream>
#include <chrono>
#include "utilities.hpp"

//' Execute a Tabu Search in the neighborhood of Solution S (Max Iterations)
//' @param \code{S} Initial solution.
//' @param \code{fitness} Initial solution fitness variable.
//' @param \code{distances} Square and symmetric distance matrix.
//' @param \code{alpha} Tenure list multiplier.
//' @param \code{rho} Neighborhood constraint coefficient.
//' @param \code{max_iteration} number of search iterations.
//' @param \code{verbose} print results
//' @returm int. interation of the best fitness. 
//' @examples
//' cnts()
// [[Rcpp::export(cnts)]]
Rcpp::List cnts(arma::uvec S,   
           double fitness, 
           const arma::mat& distances, 
           int alpha =  15, 
           double rho = 1,  
           int max_iterations = 1000,
           bool verbose = false) 
{
  
  double max_distance = distances.max(); // Max distance between two nodes
  //arma::uvec BestSoFar(S);
  //double fitness = getBinaryTourFitness(S, distanceMatrix);
  arma::uvec tabu_S(S); 
  double tabu_fitness = fitness;
  int best_iteration = 0;
  int count_iterations = 0;
  int N = distances.n_cols; // Number of nodes
  arma::uvec Tenure(N, arma::fill::zeros); // Tenure List
  // Equation 4 in http://dx.doi.org/10.1016/j.ejor.2013.06.002
  arma::vec gain = arma::sum(distances.cols(arma::find(tabu_S == 1)),1);
  while(count_iterations < max_iterations)
  {
    // Pg 544 in http://dx.doi.org/10.1016/j.ejor.2013.06.002 
    double dMinInS  = arma::min(gain.elem(arma::find(tabu_S == 1)));
    double dMaxOutS = arma::max(gain.elem(arma::find(tabu_S == 0)));
    //---------------------------------------------------------
    // Find X, Y, Delta, u and v
    arma::uvec X = (gain <=  dMinInS + rho*max_distance) % tabu_S;
    arma::uvec Y = (gain >= dMaxOutS - rho*max_distance) % (1 - tabu_S);
    // Constraint Neighbohoods
    arma::uvec CN_X = arma::find(X == 1); 
    arma::uvec CN_Y = arma::find(Y == 1);
    // Find delta matrix 
    arma::mat delta = -distances(CN_X, CN_Y);
    delta.each_col() -= gain(CN_X);
    delta.each_row() += gain(CN_Y).t();
    // find max to use in swap
    double max_delta = delta.max();
    arma::uvec max_indices = arma::shuffle(arma::find(delta == max_delta));
    //int max_rand = max_indices(arma::randi(arma::distr_param(0,max_indices.size()-1)));
    for (unsigned i = 0; i < max_indices.size(); i++){
      int u = CN_X(max_indices(i) % CN_X.size());
      int v = CN_Y(max_indices(i) / CN_X.size());
      // Swap u and v if admissible
      if (((Tenure(u) == 0) && (Tenure(v) == 0)) || 
          (fitness < (tabu_fitness + max_delta))){
        tabu_S(u) = 0;
        tabu_S(v) = 1;
        tabu_fitness += max_delta;
        gain +=  distances.col(v) - distances.col(u);
        int index = (int)std::floor((((double)(count_iterations%1500))/100.0));
        Tenure(u) = alpha*alpha_multiplier(index);
        Tenure(v) = int(0.7*Tenure(u));
        break; // leave the for
      }
    }
    if (fitness < tabu_fitness){
      S = tabu_S;
      fitness = tabu_fitness;
      best_iteration = count_iterations;
    }
    //---------------------------------------------------------
    Tenure.elem(find(Tenure > 0.5)) -= 1;
    count_iterations++;
    if (verbose) 
      Rprintf("\rIterations: %d, Best = %5.2f, Tabu = %5.2f", count_iterations, fitness, tabu_fitness);
  }
  return Rcpp::List::create(Rcpp::Named("S")          = S,
                            Rcpp::Named("fitness")    = fitness,
                            Rcpp::Named("iterations") = best_iteration);
}


//' Execute a Tabu Search in the neighborhood of Solution S (Max Iterations)
//' @param \code{S} Initial solution.
//' @param \code{fitness} Initial solution fitness variable.
//' @param \code{distances} Square and symmetric distance matrix.
//' @param \code{alpha} Tenure list multiplier.
//' @param \code{rho} Neighborhood constraint coefficient.
//' @param \code{max_iteration} number of search iterations.
//' @param \code{verbose} print results
//' @returm int. interation of the best fitness. 
//' @examples
//' cnts()
// [[Rcpp::export(cnts_sugar)]]
Rcpp::List cnts_sugar(arma::uvec S,   
                const arma::mat& distances, 
                int alpha =  15, 
                double rho = 1,  
                int max_iterations = 1000,
                bool verbose = false) 
{
  
  double max_distance = distances.max(); // Max distance between two nodes
  double fitness = tour_fitness_binary(S, distances);
  arma::uvec tabu_S(S); 
  double tabu_fitness = fitness;
  int best_iteration = 0;
  int count_iterations = 0;
  int N = distances.n_cols; // Number of nodes
  arma::uvec Tenure(N, arma::fill::zeros); // Tenure List
  // Equation 4 in http://dx.doi.org/10.1016/j.ejor.2013.06.002
  arma::vec gain = arma::sum(distances.cols(arma::find(tabu_S == 1)),1);
  while(count_iterations < max_iterations)
  {
    // Pg 544 in http://dx.doi.org/10.1016/j.ejor.2013.06.002 
    double dMinInS  = arma::min(gain.elem(arma::find(tabu_S == 1)));
    double dMaxOutS = arma::max(gain.elem(arma::find(tabu_S == 0)));
    //---------------------------------------------------------
    // Find X, Y, Delta, u and v
    arma::uvec X = (gain <=  dMinInS + rho*max_distance) % tabu_S;
    arma::uvec Y = (gain >= dMaxOutS - rho*max_distance) % (1 - tabu_S);
    // Constraint Neighbohoods
    arma::uvec CN_X = arma::find(X == 1); 
    arma::uvec CN_Y = arma::find(Y == 1);
    // Find delta matrix 
    arma::mat delta = -distances(CN_X, CN_Y);
    delta.each_col() -= gain(CN_X);
    delta.each_row() += gain(CN_Y).t();
    // find max to use in swap
    double max_delta = delta.max();
    arma::uvec max_indices = arma::shuffle(arma::find(delta == max_delta));
    //int max_rand = max_indices(arma::randi(arma::distr_param(0,max_indices.size()-1)));
    auto it = max_indices.begin();
    for (; it != max_indices.end(); ++it){
      arma::uvec idxs = arma::ind2sub(arma::size(delta), *it);
      int u = CN_X(idxs(0));
      int v = CN_Y(idxs(1));
      // Swap u and v if admissible
      if (((Tenure(u) == 0) && (Tenure(v) == 0)) || 
          (fitness < (tabu_fitness + max_delta))){
        tabu_S(u) = 0;
        tabu_S(v) = 1;
        tabu_fitness += max_delta;
        gain +=  distances.col(v) - distances.col(u);
        int index = (int)std::floor((((double)(count_iterations%1500))/100.0));
        Tenure(u) = alpha*alpha_multiplier(index);
        Tenure(v) = int(0.7*Tenure(u));
        break; // leave the for
      }
    }
    if (fitness < tabu_fitness){
      S = tabu_S;
      fitness = tabu_fitness;
      best_iteration = count_iterations;
    }
    //---------------------------------------------------------
    Tenure.elem(find(Tenure == 1)) -= 1;
    count_iterations++;
    if (verbose) 
      Rprintf("\rIterations: %d, Best = %5.2f, Tabu = %5.2f", count_iterations, fitness, tabu_fitness);
  }
  return Rcpp::List::create(Rcpp::Named("S")          = S,
                            Rcpp::Named("fitness")    = fitness,
                            Rcpp::Named("iterations") = best_iteration);
}

//' Get fitness from tour
//' @details Get fitness using the tour, m and distance matrix
//' @param \code{Tour} Set of tour's nodes.
//' @param \code{Distances} Distance matrix
//' @return A double value representing the chromosome fitness
//' @export 
// [[Rcpp::export("tour_fitness_binary")]]
double tour_fitness_binary(const arma::uvec& S, const arma::mat& distances){
  double fitness = 0;
  arma::uvec idx = arma::find(S);
  auto it1 = idx.begin();
  //unsigned n = tour_distances.n_cols;
  for (; it1 != idx.end(); it1++) {
    for (auto it2 = it1 + 1; it2 != idx.end(); it2++) {
      fitness += distances(*it1, *it2);
    }
  }
  return(fitness);
} 