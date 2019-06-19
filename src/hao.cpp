#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

#include "utilities.hpp"

/*
 * Main routines for MAMDP, OBMA and DMAMDP
 */

//' Hao's Hybrid Metaheuristic method for the Maximum Diversity Problem
//' Ref: http://dx.doi.org/10.1016/j.ejor.2013.06.002
//' @details Get over it!
//' @param \code{distances} Symmetric matrix.
//' @param \code{tour_size} Subset size of nodes.
//' @param \code{population_size} Number of individual in population.
//' @param \code{max_iterations} for the tabu search.
//' @param \code{max_time} time limit.
//' @param \code{beta} for update population.
//' @param \code{tabu_rho} for tabu search
//' @param \code{tabu_alpha} for tabu search.
//' @param \code{verbose} well.
//' @return A better person.
//' @export
// [[Rcpp::export]]
Rcpp::List hao_mamdp(const arma::mat& distances,
                     unsigned tour_size,
                     unsigned population_size = 10,
                     double max_time = 20,
                     double beta = .6,
                     unsigned population_multiplier = 3,
                     unsigned tabu_max_iterations = 50000,
                     double tabu_rho = 1,
                     double tabu_alpha = 15,
                     bool verbose = false){
  //0. Start time
  auto start = std::chrono::steady_clock::now();
  std::chrono::duration<double> diff;
  double time_elapsed = 0;
  double time_best = 0;
  //1. Initialize population
  if (verbose)
    Rprintf("\nInitializing population...");
  Rcpp::List population_list = initialize_population_mamdp(distances, tour_size, population_size, tabu_max_iterations, population_multiplier, tabu_rho, tabu_alpha);
  arma::umat population = Rcpp::as<arma::umat>(population_list["population"]);
  arma::vec population_fitness = Rcpp::as<arma::vec>(population_list["fitness"]);
  //2. Find best in population
  if (verbose)
    Rprintf("\nSorting population...");
  // int bestIndex = findBestPopFitness(population, distanceMatrix);
  arma::uvec best_solution = population.col(0);
  double best_fitness = population_fitness(0);
  //3 while loop
  if (verbose)
    Rprintf("\nStarting while loop until max time...\n");
  while (time_elapsed <= max_time){
    //4. Sample parents
    arma::uvec nodes = arma::shuffle(arma::linspace<arma::uvec>(0, population_size-1));
    arma::uvec S1 = population.col(nodes(0));
    arma::uvec S2 = population.col(nodes(1));
    //5. Crossover
    arma::uvec S0 = crossover(S1, S2, distances);
    double S0_fitness = tour_fitness_binary(S0, distances);
    //6. TabuSearch
    Rcpp::List tabu_rst = cnts(S0,  S0_fitness, distances, tabu_alpha, tabu_rho, tabu_max_iterations, false);
    if (S0_fitness > best_fitness){
      best_solution = S0;
      best_fitness = S0_fitness;
      auto time = std::chrono::steady_clock::now();
      time_best = std::chrono::duration <double, std::milli> (time - start).count()/1000;
    }
    //7. Update population
    population_list = update_population_mamdp(S0, population, population_fitness, distances, beta);
    auto time = std::chrono::steady_clock::now();
    time_elapsed = std::chrono::duration <double, std::milli> (time - start).count()/1000;
    if (verbose)
      Rprintf("Best so far: %5.0f, Current: %5.f, Best Time: %3.2f, Elapsed: %3.2f\r",
              best_fitness, S0_fitness, time_best, time_elapsed);;
    Rcpp::checkUserInterrupt();
  }
  
  Rcpp::DataFrame rst = Rcpp::DataFrame::create(
    Rcpp::Named("fitness") = best_fitness,
    Rcpp::Named("time") = time_best,
    Rcpp::Named("duration") = time_elapsed);

  return Rcpp::List::create(Rcpp::Named("tour") = best_solution,
                            Rcpp::Named("data") = rst
  );
}