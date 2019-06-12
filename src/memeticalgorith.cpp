/*
 * André Leite <leite@de.ufpe.br>
 */

#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

// C++ Libraries
#include <omp.h>
#include <sstream>
#include <chrono>


/*********************
 * Functions Headers *
 *********************/
// Scale Function used in population update
arma::vec Amamdp(arma::vec input);
// Distance between to individuals;
int getSolutionsDistance(const arma::uvec& S0, const arma::uvec& S1);
// Distance between individual and a population;
int getSolutionToPopulationDistance(const arma::uvec& S0, const arma::umat& Population);
// Distance between individual and a population by index;
int getSolutionToPopulationDistanceByIndex(int index, const arma::umat& Population);
// Get average distance between individual and its population;
int getAverageDistanceToPopulation(const arma::uvec& S0, const arma::umat& Population);
// Get average distance between individual and its population by individual index;
int getAverageDistanceToPopulationByIndex(int index, const arma::umat& Population);
// Get binary tour fitness
double getBinaryTourFitness(const arma::uvec& Tour, const arma::mat& distanceMatrix);
// Get tour fitness
double getTourFitness(const arma::uvec& Tour, const arma::mat& Distances);
// Best Fitness and solution index of population
int  findBestPopFitness(const arma::umat& Population, const arma::mat& distanceMatrix);
// Cross Over between S_a and S_b 
arma::uvec doCrossOver(const arma::uvec& S_a, const arma::uvec& S_b, const arma::mat& distanceMatrix); 
// Backbone Cross Over between S_a and S_b 
arma::umat doBackboneCrossOver(const arma::uvec& S_a, const arma::uvec& S_b, const arma::mat& distanceMatrix); 
// Execute a Tabu Search in the neighborhood of Solution S  (Parallel, MaxIter Criterium)
arma::uvec doTabuSearchParallel(arma::uvec S, const arma::mat& distanceMatrix, int alpha, double rhoOver2, int maxIterations);
// Execute a Tabu Search in the neighborhood of Solution S (MaxIter Criterium)
arma::uvec doTabuSearchMI(arma::uvec S, const arma::mat& distanceMatrix, int alpha, double rhoOver2, int maxIterations);
// Execute a Tabu Search in the neighborhood of Solution S (LostMax Criterium)
arma::uvec doTabuSearchML(arma::uvec S, const arma::mat& distanceMatrix, int alpha, double rhoOver2, int lostmaxIterations);
// Pool initialization 
arma::umat initializePool(const arma::mat& distanceMatrix, int tourSize, int populationSize, int maxIterations, int multiplier, double rhoOver2);   
// Pool initialization 
arma::umat initializePoolML(const arma::mat& distanceMatrix, int tourSize, int populationSize, int listMaxIterations, int multiplier, double rhoOver2);   
// Opposition Based Pool Initialize()
arma::umat initializeOBP(const arma::mat& distanceMatrix, int tourSize, int populationSize, int maxIterations, int multiplier, double rhoOver2);   
// update population 
arma::umat updatePopulation(arma::uvec S, arma::umat Population, const arma::mat& distanceMatrix, double beta);   
// Rank based update population 
arma::umat updatePopulationByRank(arma::uvec S, arma::umat Population, const arma::mat& distanceMatrix, double beta);
// Hao's Hybrid Metaheuristic method for the Maximum Diversity Problem
Rcpp::List  mamdp(const arma::mat& distanceMatrix, int tourSize, int populationSize, int maxIterations, double maxTime, int multiplier, double rhoOver2);   
// Hao's Opposition-based Memetic memetic search for the Maximum Diversity Problem
Rcpp::List  obma(const arma::mat& distanceMatrix, int tourSize, int populationSize, int maxIterations, double maxTime, int multiplier, double rhoOver2);   
// Hao's Diversification-driven Memetic Algorith for Maximum Diversity Problem 
Rcpp::List dmamdp(const arma::mat& distanceMatrix, int tourSize, int populationSize, double maxTime, int maxLostIterations, double p, int multiplier, double rhoOver2);   

/*****************************
 * End of  Functions Headers *
 *****************************/


/*
 * Main routines for MAMDP, OBMA and DMAMDP
 */

//' Hao's Hybrid Metaheuristic method for the Maximum Diversity Problem
//' @details Get over it!
//' @param \code{distanceMatrix} Symmetric matrix.
//' @param \code{tourSize} Subset size of nodes.
//' @param \code{populationSize} Number of individual in population.
//' @param \code{maxIterations} for the tabu search.
//' @return A better person.
//' @export 
// [[Rcpp::export]]
Rcpp::List  mamdp(const arma::mat& distanceMatrix, int tourSize, int populationSize, int maxIterations, double maxTime = 60, int multiplier = 3, double rhoOver2 = 2, bool verbose = true){
  //0. Start time
  auto start = std::chrono::steady_clock::now();
  std::chrono::duration<double> diff;
  double time_elapsed = 0;
  double time_best = 0;
  //1. Initialize population
  if (verbose) 
    Rprintf("\nInitializing population...");
  arma::umat population = initializePool(distanceMatrix, tourSize, populationSize, maxIterations, multiplier, rhoOver2);
  //2. Find best in population
  if (verbose) 
    Rprintf("\nSorting population...");
  int bestIndex = findBestPopFitness(population, distanceMatrix);
  arma::uvec bestSolution = population.col(bestIndex);
  double bestFitness = getBinaryTourFitness(bestSolution, distanceMatrix);
  //3 while loop
  if (verbose) 
    Rprintf("\nStarting while loop until max time...\n");
  while (time_elapsed <= maxTime){
    //4. Sample parents
    arma::uvec nodes = arma::shuffle(arma::linspace<arma::uvec>(0, populationSize-1));
    arma::uvec S1 = population.col(nodes(0));
    arma::uvec S2 = population.col(nodes(1));
    //5. Crossover
    arma::uvec S0 = doCrossOver(S1, S2, distanceMatrix);
    //6. TabuSearch
    S0 = doTabuSearchMI(S0, distanceMatrix, 15, rhoOver2, maxIterations);
    double S0Fitness = getBinaryTourFitness(S0, distanceMatrix);
    if (S0Fitness > bestFitness){
      bestSolution = S0;
      bestFitness = S0Fitness;
      auto time = std::chrono::steady_clock::now();
      time_best = std::chrono::duration <double, std::milli> (time - start).count()/1000;
    }
    //7. Update population
    population = updatePopulation(S0, population, distanceMatrix, .6);
    auto time = std::chrono::steady_clock::now();
    time_elapsed = std::chrono::duration <double, std::milli> (time - start).count()/1000;
    if (verbose) 
      Rprintf("Best so far: %5.0f, Current: %5.f, Best Time: %3.2f, Elapsed: %3.2f\r", \
              bestFitness, S0Fitness, time_best, time_elapsed);;
    Rcpp::checkUserInterrupt();
    
  }
  //return(bestSolution);
  Rcpp::DataFrame rst = Rcpp::DataFrame::create(
    Rcpp::Named("Fitness") = bestFitness,
    Rcpp::Named("Best") = time_best,
    Rcpp::Named("Elapsed") = time_elapsed);
  return Rcpp::List::create(Rcpp::Named("tour") = bestSolution,
                            Rcpp::Named("data") = rst
  );
}   



//' Hao's Opposition-based Memetic memetic search for the Maximum Diversity Problem
//' @details Get over it!
//' @param \code{distanceMatrix} Symmetric matrix.
//' @param \code{tourSize} Subset size of nodes.
//' @param \code{populationSize} Number of individual in population.
//' @param \code{maxIterations} for the tabu search.
//' @return A better man.
//' @export 
// [[Rcpp::export]]
Rcpp::List  obma(const arma::mat& distanceMatrix, int tourSize, int populationSize, int maxIterations, double maxTime = 60, int multiplier = 3, double rhoOver2 = 2, bool verbose = true){
  //0. Start time
  auto start = std::chrono::steady_clock::now();
  std::chrono::duration<double> diff;
  double time_elapsed = 0;
  double time_best = 0;
  //1. Initialize population
  if (verbose) 
    Rprintf("\nInitializing population...");
  arma::umat population = initializeOBP(distanceMatrix, tourSize, populationSize, maxIterations, multiplier, rhoOver2);
  //2. Find best in population
  if (verbose) 
    Rprintf("\nSorting population...");
  int bestIndex = findBestPopFitness(population, distanceMatrix);
  arma::uvec bestSolution = population.col(bestIndex);
  double bestFitness = getBinaryTourFitness(bestSolution, distanceMatrix);
  //3 while loop
  if (verbose) 
    Rprintf("\nStarting while loop until max time...\n");
  while (time_elapsed <= maxTime){
    //4. Sample parents
    arma::uvec nodes = arma::shuffle(arma::linspace<arma::uvec>(0, populationSize-1));
    arma::uvec Si = population.col(nodes(0));
    arma::uvec Sj = population.col(nodes(1));
    //5. Crossover
    arma::umat S0 = doBackboneCrossOver(Si, Sj, distanceMatrix);
    //6.1. TabuSearch
    double S0Fitness = 0;
    S0.col(0) = doTabuSearchMI(S0.col(0), distanceMatrix, 15, rhoOver2, maxIterations);
    S0Fitness = getBinaryTourFitness(S0.col(0), distanceMatrix);
    if (S0Fitness > bestFitness){
      bestSolution = S0.col(0);
      bestFitness = S0Fitness;
      auto time = std::chrono::steady_clock::now();
      time_best = std::chrono::duration <double, std::milli> (time - start).count()/1000;
    }
    //7.1. Update population
    population = updatePopulationByRank(S0.col(0), population, distanceMatrix, .6);
    //6.2. TabuSearch
    S0.col(1) = doTabuSearchMI(S0.col(1), distanceMatrix, 15, rhoOver2, maxIterations);
    S0Fitness = getBinaryTourFitness(S0.col(1), distanceMatrix);
    if (S0Fitness > bestFitness){
      bestSolution = S0.col(1);
      bestFitness = S0Fitness;
      auto time = std::chrono::steady_clock::now();
      time_best = std::chrono::duration <double, std::milli> (time - start).count()/1000;
    }
    //7.2. Update population
    population = updatePopulationByRank(S0.col(1), population, distanceMatrix, .6);
    auto time = std::chrono::steady_clock::now();
    time_elapsed = std::chrono::duration <double, std::milli> (time - start).count()/1000;
    if (verbose) 
      Rprintf("Best so far: %5.0f, Current: %5.f, Best Time: %3.2f, Elapsed: %3.2f\r", \
              bestFitness, S0Fitness, time_best, time_elapsed);
    Rcpp::checkUserInterrupt();
  }
  //return(bestSolution);
  Rcpp::DataFrame rst = Rcpp::DataFrame::create(
    Rcpp::Named("Fitness") = bestFitness,
    Rcpp::Named("Best") = time_best,
    Rcpp::Named("Elapsed") = time_elapsed);
  return Rcpp::List::create(Rcpp::Named("tour") = bestSolution,
                            Rcpp::Named("data") = rst
  );
}   


//' Hao's Diversification-driven Memetic Algorith for Maximum Diversity Problem 
//' @details Get over it!
//' @param \code{distanceMatrix} Symmetric matrix.
//' @param \code{tourSize} Subset size of nodes.
//' @param \code{populationSize} Number of individual in population.
//' @param \code{lostMaxIterations} for the tabu search.
//' @param \code{maxTime} Time limit for execution.
//' @param \code{p} Probability of get laid. Otherwise go fish in random pool.
//' @return A better man.
//' @export 
// [[Rcpp::export]]
Rcpp::List dmamdp(const arma::mat& distanceMatrix, int tourSize, int populationSize, double maxTime, int lostMaxIterations, double p, int multiplier, double rhoOver2 = 2, bool verbose = true){
  //0. Start time
  auto start = std::chrono::steady_clock::now();
  std::chrono::duration<double> diff;
  double time_elapsed = 0;
  double time_best = 0;
  //1. Initialize population
  if (verbose) 
    Rprintf("\nInitializing population...");
  arma::umat population = initializePoolML(distanceMatrix, tourSize, populationSize, lostMaxIterations, multiplier, rhoOver2);
  //2. Find best in population
  if (verbose) 
    Rprintf("\nSorting population...");
  int N = distanceMatrix.n_cols;
  int bestIndex = findBestPopFitness(population, distanceMatrix);
  arma::uvec bestSolution = population.col(bestIndex);
  double bestFitness = getBinaryTourFitness(bestSolution, distanceMatrix);
  // Rand inialization
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 rand01(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(0.0, 1.0);
  //3 while loop
  if (verbose) 
    Rprintf("\nStarting while loop until max time...\n");
  while (time_elapsed <= maxTime){
    // Candidate
    arma::uvec S0(N, arma::fill::zeros);
    if (dis(rand01) > p){
      //4. Sample parents
      arma::uvec nodes = arma::shuffle(arma::linspace<arma::uvec>(0, populationSize-1));
      arma::uvec S1 = population.col(nodes(0));
      arma::uvec S2 = population.col(nodes(1));
      //5. Crossover
      S0 = doCrossOver(S1, S2, distanceMatrix);
      //6. TabuSearch
      S0 = doTabuSearchML(S0, distanceMatrix, 15, rhoOver2, lostMaxIterations);
     } else {
      arma::uvec nodes = arma::shuffle(arma::linspace<arma::uvec>(0, N-1, N));
      arma::uvec tour = nodes.subvec(0, tourSize - 1);
      S0.elem(tour) += 1;
      S0 = doTabuSearchML(S0, distanceMatrix, 15, rhoOver2, lostMaxIterations);
      double S0Fitness = getBinaryTourFitness(S0, distanceMatrix);
      if (S0Fitness > bestFitness){
        bestSolution = S0;
        bestFitness = S0Fitness;
        auto time = std::chrono::steady_clock::now();
        time_best = std::chrono::duration <double, std::milli> (time - start).count()/1000;
      }
      population = updatePopulationByRank(S0, population, distanceMatrix, .6);
      arma::uvec randS = population.col(std::floor(populationSize*((double)dis(rand01))));
      S0 = doCrossOver(S0, randS, distanceMatrix);
      S0 = doTabuSearchML(S0, distanceMatrix, 15, rhoOver2, lostMaxIterations);
    }
    double S0Fitness = getBinaryTourFitness(S0, distanceMatrix);
    if (S0Fitness > bestFitness){
      bestSolution = S0;
      bestFitness = S0Fitness;
      auto time = std::chrono::steady_clock::now();
      time_best = std::chrono::duration <double, std::milli> (time - start).count()/1000;
    }
    //7. Update population
    population = updatePopulationByRank(S0, population, distanceMatrix, .6);
    auto time = std::chrono::steady_clock::now();
    time_elapsed = std::chrono::duration <double, std::milli> (time - start).count()/1000;
    if (verbose) 
      Rprintf("Best so far: %5.0f, Current: %5.f, Best Time: %3.2f, Elapsed: %3.2f\r", \
              bestFitness, S0Fitness, time_best, time_elapsed);
    Rcpp::checkUserInterrupt();
  }
  //return(bestSolution);
  Rcpp::DataFrame rst = Rcpp::DataFrame::create(
    Rcpp::Named("Fitness") = bestFitness,
    Rcpp::Named("Best") = time_best,
    Rcpp::Named("Elapsed") = time_elapsed);
  return Rcpp::List::create(Rcpp::Named("tour") = bestSolution,
                            Rcpp::Named("data") = rst
  );
}

