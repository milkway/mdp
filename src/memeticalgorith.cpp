#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

// C++ Libraries
#include <omp.h>
#include <sstream>
#include <chrono>


/*********************
 * Functions Headers *
 *********************/
// Get binary tour fitness (binary or not)
double getBinaryTourFitness(const arma::uvec& Tour, const arma::mat& distanceMatrix);
double getTourFitness(const arma::uvec& Tour, const arma::mat& Distances);
// Best Fitness and solution index of population
std::pair<int, double>  findBestPopFitness(const arma::umat& Population, const arma::mat& distanceMatrix);
// Cross Over between S_a and S_b 
arma::uvec doCrossOver(arma::uvec S_a, arma::uvec S_b, const arma::mat& distanceMatrix); 
// Execute a Tabu Search in the neighborhood of Solution S
arma::uvec doTabuSearch(arma::uvec S, const arma::mat& distanceMatrix, int alpha, double rhoOver2, int maxIterations);
// Pool initialization 
arma::umat initializePool(const arma::mat& distanceMatrix, int tourSize, int populationSize);   
// Opposition Based Pool Initialize()
arma::umat initializeOBP(const arma::mat& distanceMatrix, int tourSize, int populationSize);   
// update population 
arma::umat updatePopulation(arma::uvec S, arma::umat Population);   
// Hao's Memetic Algorithm for Maximum Diversity Problem
arma::uvec memeticAlgorithm(const arma::mat& distanceMatrix, int tourSize, int populationSize);   
/*****************************
 * End of  Functions Headers *
 *****************************/
  

// Best Fitness and solution index of population
std::pair<int, double>  findBestPopFitness(const arma::umat& Population, const arma::mat& distanceMatrix)
{
  std::pair <int, double> pairIndexFitness; 
  
  pairIndexFitness.first = 0; 
  pairIndexFitness.second = .1; 
  return(pairIndexFitness);
}

 
// Cross Over between S_a and S_b 
arma::uvec doCrossOver(arma::uvec S_a, arma::uvec S_b, const arma::mat& distanceMatrix) 
{
  arma::uvec Result;
  return(Result);
}


//' Execute a Tabu Search in the neighborhood of Solution S
//' @param S initial solution
//' @param distanceMatrix 
//' @return Best solution of local tabu search 
//' @examples
//' dotabuSearch()
// [[Rcpp::export]]
arma::uvec doTabuSearch(arma::uvec S, const arma::mat& distanceMatrix, int alpha = 15, double rhoOver2 = 1, int maxIterations = 100)
{
  arma::uvec alphaMultiplier = {1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1};
  double dmax = distanceMatrix.max(); // Max distance between two nodes
  arma::uvec BestSoFar(S);
  double fitness = getBinaryTourFitness(S, distanceMatrix);
  double BestFitnessSoFar = fitness; 
  int iterCount = 0;
  int N = distanceMatrix.n_cols;
  arma::uvec Tenure(N, arma::fill::zeros); // Tenure List
  while(iterCount < maxIterations)
    {
      // Find p_i, i in (1, N), and min(p_i) ---------------------
      arma::vec p = distanceMatrix*S;
      double dMinInS  = arma::min(p.elem(arma::find(S == 1)));
      double dMaxOutS = arma::max(p.elem(arma::find(S == 0)));
      //---------------------------------------------------------
      // Find X, Y, Delta, u and v
      arma::uvec X = (p <=  dMinInS + rhoOver2*dmax) % S;
      arma::uvec Y = (p >= dMaxOutS - rhoOver2*dmax) % (1 - S);
      arma::uvec NC_X = arma::find(X == 1);
      arma::uvec NC_Y = arma::find(Y == 1);
      //arma::mat delta(NC_X.size(), NC_Y.size(), arma::fill::zeros);
      // (OPENMP)
      /////
      arma::mat delta = -distanceMatrix(NC_X, NC_Y);
      delta.each_col() -= p(NC_X);
      delta.each_row() += p(NC_Y).t();
      ////
      // #pragma omp parallel for schedule(dynamic)
      // for(int j = 0; j  < NC_Y.size(); ++j)
      //   for(int i = 0; i < NC_X.size(); ++i)
      //     delta(i,j) = p(NC_Y(j)) - p(NC_X(i)) - distanceMatrix(NC_X(i),NC_Y(j));
      double bestDelta = delta.max();
      //arma::uvec max_shuffled = (arma::shuffle(arma::find(delta == bestDelta)));
      arma::uvec maxValues = arma::find(delta == bestDelta);
      int randMax = maxValues(arma::randi(arma::distr_param(0,maxValues.size()-1)));
      int u = NC_X(randMax % NC_X.size());
      int v = NC_Y(randMax / NC_X.size());
      // Swap u and x if admissible
      if (((u != v) && \
          (Tenure(u) == 0) && \
          (Tenure(v) == 0)) || \
          (BestFitnessSoFar < (fitness + bestDelta))){
        S(u) = 0;
        S(v) = 1;
        fitness += bestDelta;
        int index = (int)std::floor(15*(((double)(iterCount%1500))/1500.0));
        Tenure(u) = alpha*alphaMultiplier(index);
        Tenure(v) = int(0.7*Tenure(u));
        }
      if (BestFitnessSoFar < fitness){
        BestSoFar = S;
        BestFitnessSoFar = fitness;
      }
      //---------------------------------------------------------
      Tenure.elem( find(Tenure > 0.5) ) -= 1;
      iterCount++;
      //Rprintf("\nBest = %3.2f, S = %3.2f", BestFitnessSoFar, fitness);
    }
  return(BestSoFar);  
}

// Execute a Tabu Search in the neighborhood of Solution S
arma::umat updatePopulation(arma::uvec S, arma::umat Population)
{
  arma::umat newPopulation; 
  return(newPopulation); 
}


/*
 * Hao's Memetic Algorithm for Maximum Diversity Problem 
 */
// [[Rcpp::export]]
arma::uvec memeticAlgorithm(const arma::mat& distanceMatrix, int tourSize, int populationSize)   
{
  arma::uvec Tour;
  return(Tour);
}
 
 //' Opposition Based Population initialization 
 //' @details Get the initial populatio
 //' @param \code{Tour Size}.
 //' @param \code{Distances} Distance matrix
 //' @param \code{Population Size} 
 //' @return A matrix where each column is a individual in the population
 //' @export 
 // [[Rcpp::export("initializeOPB")]] 
 arma::umat initializeOBP(const arma::mat& distanceMatrix, int tourSize, int populationSize = 10){
   int N = distanceMatrix.n_cols;
   arma::umat population(N, populationSize, arma::fill::zeros);
   if (tourSize > N/2) 
     Rcpp::stop("Not implemented yet. Try with tour size smaller than N/2.");
  int count = 0; 
  // while(count < populationSize){
  //   arma::uvec nodesNeverUsed = arma::find(arma::sum(population,1) == 0);
  // }
  // if (nodesNeverUsed.size() >= tourSize){
  // 
  // }

  return(population);    
 }   
 
 //' Get fitness from tour
 //' @details Get fitness using the tour, m and distance matrix
 //' @param \code{Tour} Set of tour's nodes.
 //' @param \code{Distances} Distance matrix
 //' @return A double value representing the chromosome fitness
 //' @export 
 // [[Rcpp::export("getBinaryTourFitness")]]
 double getBinaryTourFitness(const arma::uvec& Tour, const arma::mat& distanceMatrix){
   double Fitness = 0;
     unsigned n = Tour.size();
   for (unsigned i = 0; i < n; i++) {
     for (auto j = i + 1; j <  n; j++) {
       Fitness += Tour(i)*Tour(j)*distanceMatrix(i, j);
     }
   }
   return(Fitness);
 } 

//' Get fitness from tour
//' @details Get fitness using the tour, m and distance matrix
//' @param \code{Tour} Set of tour's nodes.
//' @param \code{Distances} Distance matrix
//' @return A double value representing the chromosome fitness
//' @export 
// [[Rcpp::export("getTourFitness")]]
double getTourFitness(const arma::uvec& Tour, const arma::mat& distanceMatrix){
  double Fitness = 0;
  unsigned m = Tour.size();
  for (unsigned i = 0; i < m; i++) {
    for (auto j = i; j <  m; j++) {
      Fitness += distanceMatrix(Tour(i), Tour(j));
    }
  }
  return(Fitness);
}