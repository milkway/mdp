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




//' Best Fitness and solution index of population
//' @param \code{Population} to fit.
//' @param \code{distanceMatrix} Square and symmetric distance matrix 
//' @return A baby 
//' @examples
//' findBestPopFitness()
// [[Rcpp::export]]
int  findBestPopFitness(const arma::umat& Population, const arma::mat& distanceMatrix){
  int P = Population.n_cols;
  arma::vec fitness(P, arma::fill::zeros);
  for(int i = 0; i < P; i++)
    fitness(i) = getBinaryTourFitness(Population.col(i), distanceMatrix);
  return(arma::index_max(fitness));
}


//' Cross Over between S_a and S_b 
//' @param \code{S_a} Parent A
//' @param \code{S_b} Parent B
//' @param \code{distanceMatrix} Square and symmetric distance matrix 
//' @return A baby 
//' @examples
//' doCrossOver()
// [[Rcpp::export]]
arma::uvec doCrossOver(const arma::uvec& S_a, const arma::uvec& S_b, const arma::mat& distanceMatrix) 
{
  arma::uvec child(S_a % S_b); // Backbone
  arma::uvec poolA = arma::find((S_a + child) == 1); // The other poor little guys in S_a
  arma::uvec poolB = arma::find((S_b + child) == 1); // The other poor little guys in S_b
  int m = arma::sum(S_a);
  int order = 0;
  int goingIn = 0;
  while(true){
    if (poolA.size() > 0){
      arma::mat deltaA = distanceMatrix(arma::find(child == 1), poolA);
      order = arma::index_max(arma::sum(deltaA, 0));
      goingIn = poolA(order);
      child(goingIn) = 1;
      poolA.shed_row(order);
    }
    if (arma::sum(child) >= m) break;
    if (poolB.size() > 0){
      arma::mat deltaB = distanceMatrix(arma::find(child == 1), poolB);
      order = arma::index_max(arma::sum(deltaB, 0));
      goingIn = poolB(order);
      child(goingIn) = 1;
      poolB.shed_row(order); 
    }
    if (arma::sum(child) >= m) break;
  }
  //Rcpp::Rcout << "S0sum "<< arma::sum(child) << std::endl;
  return(child);
}

//' Backbone Cross Over between S_a and S_b 
//' @param \code{S_a} Parent A
//' @param \code{S_b} Parent B
//' @param  \code{distanceMatrix} Square and symmetric distance matrix 
//' @return A baby 
//' @examples
//' doBackboneCrossOver()
// [[Rcpp::export]]
arma::umat doBackboneCrossOver(const arma::uvec& S_a, const arma::uvec& S_b, const arma::mat& distanceMatrix){
  arma::uvec child(S_a % S_b); // Backbone
  arma::uvec pool = arma::find((S_a + S_b) == 1); // The other poor little guys
  //arma::mat delta = distanceMatrix.cols(pool);
  int m = arma::sum(S_a);
  while(arma::sum(child) < m){
    arma::mat delta = distanceMatrix(arma::find(child == 1), pool);
    int order = arma::index_max(arma::sum(delta, 0));
    int goingIn = pool(order);
    child(goingIn) = 1;
    pool.shed_row(order);
  }
  arma::uvec antiChild(S_a % S_b); // Backbone
  arma::uvec antiPool = arma::find((S_a + S_b + child) == 1); // The other very poor little guys
  //arma::mat delta = distanceMatrix.cols(pool);
  while(arma::sum(antiChild) < m){
    arma::mat delta = distanceMatrix(arma::find(antiChild == 1), antiPool);
    int order = arma::index_max(arma::sum(delta, 0));
    int goingIn = antiPool(order);
    antiChild(goingIn) = 1;
    antiPool.shed_row(order);
  }
  arma::umat childs(S_a.size(),2, arma::fill::zeros);
  childs.col(0) = child;
  childs.col(1) = antiChild;
  return(childs);
} 


//' Execute a Tabu Search in the neighborhood of Solution S (Max Iterations)
//' @param \code{S} initial solution
//' @param \code{distanceMatrix} Square and symmetric distance matrix
//' @return Best solution of local tabu search 
//' @examples
//' dotabuSearch()
// [[Rcpp::export]]
arma::uvec doTabuSearchMI(arma::uvec S, const arma::mat& distanceMatrix, int alpha = 15, double rhoOver2 = 2, int maxIterations = 1000)
{
  arma::uvec alphaMultiplier = {1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1};
  double dmax = distanceMatrix.max(); // Max distance between two nodes
  arma::uvec BestSoFar(S);
  double fitness = getBinaryTourFitness(S, distanceMatrix);
  double BestFitnessSoFar = fitness; 
  int iterCount = 0;
  int N = distanceMatrix.n_cols;
  arma::uvec Tenure(N, arma::fill::zeros); // Tenure List
  arma::vec p = arma::sum(distanceMatrix.cols(arma::find(S == 1)),1);
  while(iterCount < maxIterations)
  {
    // Find p_i, i in (1, N), and min(p_i) ---------------------
    //arma::vec p = distanceMatrix*S;
    //arma::vec p = arma::sum(distanceMatrix.cols(arma::find(S == 1)),1);
    double dMinInS  = arma::min(p.elem(arma::find(S == 1)));
    double dMaxOutS = arma::max(p.elem(arma::find(S == 0)));
    //---------------------------------------------------------
    // Find X, Y, Delta, u and v
    arma::uvec X = (p <=  dMinInS + rhoOver2*dmax) % S;
    arma::uvec Y = (p >= dMaxOutS - rhoOver2*dmax) % (1 - S);
    arma::uvec NC_X = arma::find(X == 1);
    arma::uvec NC_Y = arma::find(Y == 1);
    // arma::mat delta(NC_X.size(), NC_Y.size(), arma::fill::zeros);
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
    if (((u != v) &&         \
        (Tenure(u) == 0) &&  \
        (Tenure(v) == 0)) || \
        (BestFitnessSoFar < (fitness + bestDelta))){
      S(u) = 0;
      S(v) = 1;
      p +=  distanceMatrix.col(v) - distanceMatrix.col(u);
      double dp = distanceMatrix(v, u);
      //p(u) += dp;
      //p(v) -= dp;
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

//' Execute a Tabu Search in the neighborhood of Solution S (Max Losts)
//' @param \code{S} initial solution
//' @param \code{distanceMatrix} Square and symmetric distance matrix
//' @return Best solution of local tabu search 
//' @examples
//' dotabuSearch()
// [[Rcpp::export]]
arma::uvec doTabuSearchML(arma::uvec S, const arma::mat& distanceMatrix, int alpha = 15, double rhoOver2 = 1, int lostMaxIterations = 1000)
{
  arma::uvec alphaMultiplier = {1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1};
  double dmax = distanceMatrix.max(); // Max distance between two nodes
  arma::uvec BestSoFar(S);
  double fitness = getBinaryTourFitness(S, distanceMatrix);
  double BestFitnessSoFar = fitness; 
  int lostIterCount = 0;
  int N = distanceMatrix.n_cols;
  arma::uvec Tenure(N, arma::fill::zeros); // Tenure List
  arma::vec p = arma::sum(distanceMatrix.cols(arma::find(S == 1)),1);
  while(lostIterCount < lostMaxIterations)
  {
    // Find p_i, i in (1, N), and min(p_i) ---------------------
    //arma::vec p = distanceMatrix*S;
    //arma::vec p = arma::sum(distanceMatrix.cols(arma::find(S == 1)),1);
    double dMinInS  = arma::min(p.elem(arma::find(S == 1)));
    double dMaxOutS = arma::max(p.elem(arma::find(S == 0)));
    //---------------------------------------------------------
    // Find X, Y, Delta, u and v
    arma::uvec X = (p <=  dMinInS + rhoOver2*dmax) % S;
    arma::uvec Y = (p >= dMaxOutS - rhoOver2*dmax) % (1 - S);
    arma::uvec NC_X = arma::find(X == 1);
    arma::uvec NC_Y = arma::find(Y == 1);
    // arma::mat delta(NC_X.size(), NC_Y.size(), arma::fill::zeros);
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
    if (((u != v) &&         \
        (Tenure(u) == 0) &&  \
        (Tenure(v) == 0)) || \
        (BestFitnessSoFar < (fitness + bestDelta))){
      S(u) = 0;
      S(v) = 1;
      p +=  distanceMatrix.col(v) - distanceMatrix.col(u);
      double dp = distanceMatrix(v, u);
      //p(u) += dp;
      //p(v) -= dp;
      fitness += bestDelta;
      int index = (int)std::floor(15*(((double)(lostIterCount%1500))/1500.0));
      Tenure(u) = alpha*alphaMultiplier(index);
      Tenure(v) = int(0.7*Tenure(u));
    }
    if (BestFitnessSoFar < fitness){
      BestSoFar = S;
      BestFitnessSoFar = fitness;
      lostIterCount = 0;
    } else {
      lostIterCount++;
    }
    //---------------------------------------------------------
    Tenure.elem( find(Tenure > 0.5) ) -= 1;
    //Rprintf("\nBest = %3.2f, S = %3.2f", BestFitnessSoFar, fitness);
  }
  return(BestSoFar);  
}


//' MAMDP population update
//' @param \code{S} population candidate
//' @param \code{Population} population to be updated
//' @return New population 
//' @examples
//' updatePopulation()
// [[Rcpp::export(updatePopulationMAMDP)]]
arma::umat updatePopulation(arma::uvec S, arma::umat Population, const arma::mat& distanceMatrix, double beta = 0.6){
  int P = Population.n_cols;
  int N = S.size();
  arma::umat newPopulation(N, P + 1, arma::fill::zeros);
  newPopulation.cols(0, P-1) = Population;
  newPopulation.col(P) = S;
  arma::vec fitness(P+1, arma::fill::zeros);
  arma::vec distance(P+1, arma::fill::zeros);
  for(int i = 0; i < P+1; i++){
    fitness(i) = getBinaryTourFitness(newPopulation.col(i), distanceMatrix);
    distance(i) = getSolutionToPopulationDistanceByIndex(i, newPopulation);
  }
  arma::vec score = beta*(Amamdp(fitness)) + (1-beta)*(Amamdp(distance));
  newPopulation.shed_col(arma::index_min(score));
  return(newPopulation); 
}

//Normalize Auxiliary function
arma::vec Amamdp(arma::vec input){
  double max_in = arma::max(input);
  double min_in = arma::min(input);
  return((input - min_in)/(max_in - min_in + 1));
}


//' Rank based pool population update
//' @param \code{S} population candidate
//' @param \code{Population} population to be updated
//' @return New population 
//' @examples
//' updatePopulationByRank()
// [[Rcpp::export]]
arma::umat updatePopulationByRank(arma::uvec S, arma::umat Population, const arma::mat& distanceMatrix, double beta = 0.6){
  int P = Population.n_cols;
  int N = S.size();
  arma::umat newPopulation(N, P + 1, arma::fill::zeros);
  newPopulation.cols(0, P-1) = Population;
  newPopulation.col(P) = S;
  arma::vec fitness(P+1, arma::fill::zeros);
  arma::vec averageDistance(P+1, arma::fill::zeros);
  for(int i = 0; i < P+1; i++){
    fitness(i) = getBinaryTourFitness(newPopulation.col(i), distanceMatrix);
    averageDistance(i) = getAverageDistanceToPopulationByIndex(i, newPopulation);
  }
  arma::vec score = beta*(arma::conv_to<arma::vec>::from(arma::sort_index(fitness))) + \
    (1-beta)*(arma::conv_to<arma::vec>::from(arma::sort_index(averageDistance)));
  newPopulation.shed_col(arma::index_min(score));
  return(newPopulation); 
}


//' Opposition Based Population initialization 
//' @details Get the initial populatio
//' @param \code{TourSize} Well..
//' @param \code{Distances} Distance matrix
//' @param \code{PopulationSize} That's it.
//' @return A matrix where each column is a individual in the population
//' @export 
// [[Rcpp::export("initializeOPB")]] 
arma::umat initializeOBP(const arma::mat& distanceMatrix, int tourSize, int populationSize = 10, int maxIterations = 100, int multiplier = 1, double rhoOver2 = 2){
  int N = distanceMatrix.n_cols;
  if (tourSize > N/2) 
    Rcpp::stop("Not implemented yet. Try with tour size smaller than N/2.");
  int count = 0;
  arma::umat population(N, populationSize, arma::fill::zeros);
  int flag = 0;
  while(count < populationSize){
    flag++;
    int Limit = multiplier * arma::sum(arma::sum(population, 0) == 0);
    arma::umat macroPopulation(N, Limit, arma::fill::zeros);
    arma::vec populationFitness(Limit, arma::fill::zeros);
#pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < Limit; i++){
      arma::uvec nodes = arma::shuffle(arma::linspace<arma::uvec>(0, N-1, N));
      arma::uvec frontTour = nodes.subvec(0, tourSize - 1);
      arma::uvec rearTour  = nodes.subvec(N - tourSize, N - 1);
      arma::uvec frontCandidate(N, arma::fill::zeros);
      arma::uvec  rearCandidate(N, arma::fill::zeros);
      frontCandidate.elem(frontTour) += 1;
      rearCandidate.elem(rearTour)  += 1;
      frontCandidate = doTabuSearchMI(frontCandidate, distanceMatrix, 15, rhoOver2, maxIterations);
      rearCandidate  = doTabuSearchMI(rearCandidate, distanceMatrix, 15, rhoOver2, maxIterations);
      double frontFitness = getBinaryTourFitness(frontCandidate, distanceMatrix);
      double rearFitness = getBinaryTourFitness(rearCandidate, distanceMatrix);
      if (frontFitness >= rearFitness){
        macroPopulation.col(i) = frontCandidate;
        populationFitness(i) = frontFitness;
      } else {
        macroPopulation.col(i) = rearCandidate; 
        populationFitness(i) = rearFitness;
      }
    }
    arma::uvec best_index = sort_index(populationFitness, "descend");
    auto it = best_index.begin(); 
    if (count == 0) {
      population.col(0) =  macroPopulation.col(best_index(0));
      it++;
      count++;
    }  
    
    for(; it != best_index.end(); it++){
      int distance = getSolutionToPopulationDistance(macroPopulation.col(*it), population.cols(0, count - 1));
      if (distance >= 1) {
        population.col(count) = macroPopulation.col(*it);
        count++;
        if (count == populationSize) break;
      }
    }  
    
    if (flag > 10*populationSize) {
      Rcpp::Rcout << "Houston, we have a problem. Not that much diversity." << std::endl; 
      break;
    }
  }
  return(population);
}   

//' Pool Population initialization 
//' @details Get the initial populatio
//' @param \code{tourSize} Ok. Stop it.
//' @param \code{Distances} Distance matrix
//' @param \code{populationSize} You know! 
//' @return A matrix where each column is a individual in the population
//' @export 
// [[Rcpp::export("initializePool")]] 
arma::umat initializePool(const arma::mat& distanceMatrix, int tourSize, int populationSize = 10, int maxIterations = 100, int multiplier = 1, double rhoOver2 = 2){
  int N = distanceMatrix.n_cols;
  // if (!distanceMatrix.is_square())
  //   Rcpp::stop("Distance Matrix must be square");
  int count = 0;
  arma::umat population(N, populationSize, arma::fill::zeros);
  int flag = 0;
  while(count < populationSize){
    flag++;
    int Limit = multiplier * arma::sum(arma::sum(population, 0) == 0);
    arma::umat macroPopulation(N, Limit, arma::fill::zeros);
    arma::vec populationFitness(Limit, arma::fill::zeros);
#pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < Limit; i++){
      arma::uvec nodes = arma::shuffle(arma::linspace<arma::uvec>(0, N-1, N));
      arma::uvec tour = nodes.subvec(0, tourSize - 1);
      arma::uvec candidate(N, arma::fill::zeros);
      candidate.elem(tour) += 1;
      macroPopulation.col(i) = doTabuSearchMI(candidate, distanceMatrix, 15, rhoOver2, maxIterations);
      populationFitness(i) = getBinaryTourFitness(candidate, distanceMatrix);
    }
    arma::uvec best_index = sort_index(populationFitness, "descend");
    auto it = best_index.begin(); 
    if (count == 0) {
      population.col(0) =  macroPopulation.col(best_index(0));
      it++;
      count++;
    }  
    
    for(; it != best_index.end(); it++){
      int distance = getSolutionToPopulationDistance(macroPopulation.col(*it), population.cols(0, count - 1));
      if (distance >= 1) {
        population.col(count) = macroPopulation.col(*it);
        count++;
        if (count == populationSize) break;
      }
    }  
    
    if (flag >  10*populationSize) {
      Rcpp::Rcout << "Houston, we have a problem. Not that much diversity." << std::endl; 
      break;
    }
  }
  return(population);
}

//' Pool Population initialization (Maximum Lost)
//' @details Get the initial populatio
//' @param \code{TourSize} Thats it.
//' @param \code{Distances} Distance matrix
//' @param \code{populationSize} Same old guy 
//' @return A matrix where each column is a individual in the population
//' @export 
// [[Rcpp::export]] 
arma::umat initializePoolML(const arma::mat& distanceMatrix, int tourSize, int populationSize = 10, int lostMaxIterations = 100, int multiplier = 1, double rhoOver2 = 2){
  int N = distanceMatrix.n_cols;
  // if (!distanceMatrix.is_square())
  //   Rcpp::stop("Distance Matrix must be square");
  int count = 0;
  arma::umat population(N, populationSize, arma::fill::zeros);
  int flag = 0;
  while(count < populationSize){
    flag++;
    int Limit = multiplier * arma::sum(arma::sum(population, 0) == 0);
    arma::umat macroPopulation(N, Limit, arma::fill::zeros);
    arma::vec populationFitness(Limit, arma::fill::zeros);
#pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < Limit; i++){
      arma::uvec nodes = arma::shuffle(arma::linspace<arma::uvec>(0, N-1, N));
      arma::uvec tour = nodes.subvec(0, tourSize - 1);
      arma::uvec candidate(N, arma::fill::zeros);
      candidate.elem(tour) += 1;
      macroPopulation.col(i) = doTabuSearchML(candidate, distanceMatrix, 15, rhoOver2, lostMaxIterations);
      populationFitness(i) = getBinaryTourFitness(candidate, distanceMatrix);
    }
    arma::uvec best_index = sort_index(populationFitness, "descend");
    auto it = best_index.begin(); 
    if (count == 0) {
      population.col(0) =  macroPopulation.col(best_index(0));
      it++;
      count++;
    }  
    
    for(; it != best_index.end(); it++){
      int distance = getSolutionToPopulationDistance(macroPopulation.col(*it), population.cols(0, count - 1));
      if (distance >= 1) {
        population.col(count) = macroPopulation.col(*it);
        count++;
        if (count == populationSize) break;
      }
    }  
    
    if (flag >  10*populationSize) {
      Rcpp::Rcout << "Houston, we have a problem. Not that much diversity." << std::endl; 
      break;
    }
  }
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
#pragma omp parallel for reduction(+: Fitness)
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

//' Distance between to individuals;
//' @details Distance between to individuals: m - sum(S0XS1);
//' @param \code{S0} Individual.
//' @param \code{S1} Individual.
//' @return A int, number of moves to go from one individual to the other.
//' @export 
// [[Rcpp::export]]
int getSolutionsDistance(const arma::uvec& S0, const arma::uvec& S1){
  int m = arma::sum(S0);
  if (m != arma::sum(S1))
    Rcpp::stop("Tour sizes must be iguals");
  return(m - arma::sum(S0 % S1));
}

//' Distance between a individual and a population;
//' @details Distance between to individuals in population: min(m - sum(S0XS), S in Population);
//' @param \code{S0} Individual.
//' @param \code{Population} Target Population.
//' @return A int, min distance between a indivitual and a population.
//' @export 
// [[Rcpp::export]]
int getSolutionToPopulationDistance(const arma::uvec& S0, const arma::umat& Population){
  int m = arma::sum(S0);
  auto M(Population);
  if (any(arma::sum(M,0) - m != 0 ))
    Rcpp::stop("Tour sizes must be iguals");
  M.each_col() %= S0; // Multiply each individual by S0, element wise.  
  return(arma::min(m - arma::sum(M,0)));
}


//' Average distance between a individual and its population;
//' @details Average distance between a individual and other elements in population: mean(m - sum(S0XS), S in Population, S not i);
//' @param \code{S0} Individual.
//' @param \code{Population} Target Population.
//' @return A int, min distance between a indivitual and a population.
//' @export 
// [[Rcpp::export]]
int getAverageDistanceToPopulation(const arma::uvec& S0, const arma::umat& Population){
  int m = arma::sum(S0);
  auto M(Population);
  if (any(arma::sum(M,0) - m != 0 ))
    Rcpp::stop("Tour sizes must be iguals");
  M.each_col() %= S0; // Multiply each individual by S0, element wise.  
  return((arma::sum(m - arma::sum(M,0)))/(Population.n_cols - 1));
}

//' Average distance between a individual and its population (by index);
//' @details Average distance between a individual and other elements in population: mean(m - sum(S0XS), S in Population, S not i);
//' @param \code{S0} Individual.
//' @param \code{Population} Target Population.
//' @return A int, min distance between a indivitual and a population.
//' @export 
// [[Rcpp::export]]
int getAverageDistanceToPopulationByIndex(int index, const arma::umat& Population){
  int m = arma::sum(Population.col(0));
  if (!((index >= 0)&&(index < Population.n_cols)))
    Rcpp::stop("Sorry. Here we count from 0 to N-1");
  auto M(Population);
  M.each_col() %= M.col(index); // Multiply each individual by S0, element wise.  
  return((arma::sum(m - arma::sum(M,0)))/(Population.n_cols - 1));
}

//' Distance between a individual and a population by index;
//' @details Distance between to individuals in population: min(m - sum(S0XS), S in Population);
//' @param \code{S0} Individual.
//' @param \code{Population} Target Population.
//' @return A int, min distance between a indivitual and a population.
//' @export 
// [[Rcpp::export]]
int getSolutionToPopulationDistanceByIndex(int index, const arma::umat& Population){
  int m = arma::sum(Population.col(0));
  if (!((index >= 0)&&(index < Population.n_cols)))
    Rcpp::stop("Sorry. Here we count from 0 to N-1");
  auto M(Population);
  M.each_col() %= M.col(index); // Multiply each individual by S0, element wise.  
  M.shed_col(index);
  return(arma::min(m - arma::sum(M,0)));
}


//' Execute a Tabu Search in the neighborhood of Solution S (Max Iterations)
//' @param \code{S} initial solution
//' @param \code{distanceMatrix} Square and symmetric distance matrix
//' @return Best solution of local tabu search 
//' @examples
//' dotabuSearch()
// [[Rcpp::export]]
arma::uvec doTabuSearchParallel(arma::uvec S, const arma::mat& distanceMatrix, int alpha = 15, double rhoOver2 = 4, int maxIterations = 1000)
{
  auto Start = std::chrono::steady_clock::now();
  
  arma::uvec alphaMultiplier = {1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1};
  double dmax = distanceMatrix.max(); // Max distance between two nodes
  arma::uvec BestSoFar(S);
  double fitness = getBinaryTourFitness(S, distanceMatrix);
  double BestFitnessSoFar = fitness; 
  int iterCount = 0;
  int N = distanceMatrix.n_cols;
  int M = sum(S);
  arma::uvec Tenure(N, arma::fill::zeros); // Tenure List
  double t0 = 0;
  double t1 = 0;
  double t2 = 0;
  double t3 = 0;
  double t4 = 0; 
  double t5 = 0;
  double t6 = 0;
  double total = 0;
  arma::vec p = arma::sum(distanceMatrix.cols(arma::find(S == 1)),1);
  while(iterCount < maxIterations)
  {
    auto start_while = std::chrono::steady_clock::now();
    
    // Find p_i, i in (1, N), and min(p_i) ---------------------
    //arma::vec p = arma::sum(distanceMatrix.cols(arma::find(S == 1)),1);
    
    auto find_p = std::chrono::steady_clock::now();
    
    double dMinInS  = arma::min(p.elem(arma::find(S == 1)));
    double dMaxOutS = arma::max(p.elem(arma::find(S == 0)));
    //---------------------------------------------------------
    
    auto find_dMs = std::chrono::steady_clock::now();
    
    
    // Find X, Y, Delta, u and v
    arma::uvec X = (p <=  dMinInS + rhoOver2*dmax) % S;
    arma::uvec Y = (p >= dMaxOutS - rhoOver2*dmax) % (1 - S);
    arma::uvec NC_X = arma::find(X == 1);
    arma::uvec NC_Y = arma::find(Y == 1);
    //arma::mat delta(NC_X.size(), NC_Y.size(), arma::fill::zeros);
    
    auto find_NC = std::chrono::steady_clock::now();
    
    // (OPENMP)
    /////
    arma::mat delta = -distanceMatrix(NC_X, NC_Y);
    delta.each_col() -= p(NC_X);
    delta.each_row() += p(NC_Y).t();
    ////
    // #pragma omp parallel for schedule(dynamic, 1)
    // for(int j = 0; j  < NC_Y.size(); ++j)
    //   for(int i = 0; i < NC_X.size(); ++i)
    //     delta(i,j) = p(NC_Y(j)) - p(NC_X(i)) - distanceMatrix(NC_X(i),NC_Y(j));
    double bestDelta = delta.max();
    
    auto find_delta = std::chrono::steady_clock::now();
    
    
    //arma::uvec max_shuffled = (arma::shuffle(arma::find(delta == bestDelta)));
    arma::uvec maxValues = arma::find(delta == bestDelta);
    //Rcpp::Rcout << "Delta" << delta;
    //Rcpp::Rcout << "maxValues"<< maxValues;
    //Rcpp::Rcout << "NC_X" << NC_X;
    //Rcpp::Rcout << "NC_Y" << NC_Y;
    int randMax = maxValues(arma::randi(arma::distr_param(0,maxValues.size()-1)));
    int u = NC_X(randMax % NC_X.size());
    int v = NC_Y(randMax / NC_X.size());
    //Rcpp::Rcout << "u " << u << " v " << v << std::endl;
    //Rcpp::Rcout << "Tu " << Tenure(u) << " Tv " << Tenure(v) << std::endl;
    auto find_uv = std::chrono::steady_clock::now();
    
    // Swap u and x if admissible
    if (((u != v) &&         \
        (Tenure(u) == 0) &&  \
        (Tenure(v) == 0)) || \
        (BestFitnessSoFar < (fitness + bestDelta))){
      S(u) = 0;
      S(v) = 1;
      p +=  distanceMatrix.col(v) - distanceMatrix.col(u);
      //double dp = distanceMatrix(v, u);
      //p(u) += dp;
      //p(v) -= dp;
      fitness += bestDelta;
      int index = (int)std::floor(15*(((double)(iterCount%1500))/1500.0));
      Tenure(u) = alpha*alphaMultiplier(index);
      Tenure(v) = int(0.7*Tenure(u));
    }
    
    auto swap_uv = std::chrono::steady_clock::now();
    
    if (BestFitnessSoFar < fitness){
      BestSoFar = S;
      BestFitnessSoFar = fitness;
      Rprintf("\nBest = %5.1f, S = %5.1f, Delta = %5.2f, Count = %d\n", BestFitnessSoFar, fitness, bestDelta, iterCount);
    }
    //---------------------------------------------------------
    Tenure.elem( find(Tenure > 0.5) ) -= 1;
    iterCount++;
    
    auto tenure_and_fitness = std::chrono::steady_clock::now();
    t0 += std::chrono::duration <double, std::milli> (find_p - start_while).count()/1000;
    t1 += std::chrono::duration <double, std::milli> (find_dMs - find_p).count()/1000;
    t2 += std::chrono::duration <double, std::milli> (find_NC - find_dMs).count()/1000;
    t3 += std::chrono::duration <double, std::milli> (find_delta - find_NC).count()/1000;
    t4 += std::chrono::duration <double, std::milli> (find_uv - find_delta).count()/1000;
    t5 += std::chrono::duration <double, std::milli> (swap_uv - find_uv).count()/1000;
    t6 += std::chrono::duration <double, std::milli> (tenure_and_fitness - swap_uv).count()/1000;
    //Rprintf("                                                                            \r");
    //Rprintf("\nBest = %5.1f, S = %5.1f, Delta = %5.2f, Count = %d\n", BestFitnessSoFar, fitness, bestDelta, iterCount);
  }
  auto End = std::chrono::steady_clock::now();
  total = std::chrono::duration <double, std::milli> (End - Start).count()/1000;
  Rprintf("\n T0 = %.3f, \n T1 = %.3f, \n T2 = %.3f, \n T3 = %.3f, \n T4 = %.3f, \n T5 = %.3f, \n T6 = %.3f, \n Total = %.3f", t0, t1, t2, t3, t4, t5, t6, total);
  return(BestSoFar);  
}



//' Pool Population initialization 
//' @details Get the initial populatio
//' @param \code{tourSize} Ok. Stop it.
//' @param \code{Distances} Distance matrix
//' @param \code{populationSize} You know! 
//' @return A matrix where each column is a individual in the population
//' @export 
// [[Rcpp::export("initializeTiming")]] 
arma::umat initializePoolTiming(const arma::mat& distanceMatrix, int tourSize, int populationSize = 10, int maxIterations = 100, int multiplier = 1){
  auto Start = std::chrono::steady_clock::now();
  int N = distanceMatrix.n_cols;
  // if (!distanceMatrix.is_square())
  //   Rcpp::stop("Distance Matrix must be square");
  int count = 0;
  arma::umat population(N, populationSize, arma::fill::zeros);
  int flag = 0;
  double t0 = 0;
  double t1 = 0;
  double t2 = 0;
  double t3 = 0;
  double t4 = 0; 
  double t5 = 0;
  double t6 = 0;
  double total = 0;
  while(count < populationSize){
    auto start_while = std::chrono::steady_clock::now();
    flag++;
    int Limit = multiplier * arma::sum(arma::sum(population, 0) == 0);
    arma::umat macroPopulation(N, Limit, arma::fill::zeros);
    arma::vec populationFitness(Limit, arma::fill::zeros);
    
    auto setup = std::chrono::steady_clock::now();
    
#pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < Limit; i++){
      arma::uvec nodes = arma::shuffle(arma::linspace<arma::uvec>(0, N-1, N));
      arma::uvec tour = nodes.subvec(0, tourSize - 1);
      arma::uvec candidate(N, arma::fill::zeros);
      candidate.elem(tour) += 1;
      macroPopulation.col(i) = doTabuSearchMI(candidate, distanceMatrix, 15, 1, maxIterations);
      populationFitness(i) = getBinaryTourFitness(candidate, distanceMatrix);
    }
    auto parallel_for = std::chrono::steady_clock::now();
    
    arma::uvec best_index = sort_index(populationFitness, "descend");
    auto it = best_index.begin(); 
    if (count == 0) {
      population.col(0) =  macroPopulation.col(best_index(0));
      it++;
      count++;
    }
    auto count_sol = std::chrono::steady_clock::now();
    
    for(; it != best_index.end(); it++){
      int distance = getSolutionToPopulationDistance(macroPopulation.col(*it), population.cols(0, count - 1));
      if (distance >= 1) {
        population.col(count) = macroPopulation.col(*it);
        count++;
        if (count == populationSize) break;
      }
    }  
    
    auto pop_size = std::chrono::steady_clock::now();
    
    if (flag >  10*populationSize) {
      Rcpp::Rcout << "Houston, we have a problem. Not that much diversity." << std::endl; 
      break;
    }
    
    t0 += std::chrono::duration <double, std::milli> (setup - start_while).count()/1000;
    t1 += std::chrono::duration <double, std::milli> (parallel_for - setup).count()/1000;
    t2 += std::chrono::duration <double, std::milli> (count_sol - parallel_for).count()/1000;
    t3 += std::chrono::duration <double, std::milli> (pop_size - count_sol).count()/1000;
    //t4 += std::chrono::duration <double, std::milli> (find_uv - find_delta).count()/1000;
    //t5 += std::chrono::duration <double, std::milli> (swap_uv - find_uv).count()/1000;
    //t6 += std::chrono::duration <double, std::milli> (tenure_and_fitness - swap_uv).count()/1000;
  }
  auto End = std::chrono::steady_clock::now();
  total = std::chrono::duration <double, std::milli> (End - Start).count()/1000;
  Rprintf("\n T0 = %.3f, \n T1 = %.3f, \n T2 = %.3f, \n T3 = %.3f, \n T4 = %.3f, \n T5 = %.3f, \n T6 = %.3f, \n Total = %.3f", t0, t1, t2, t3, t4, t5, t6, total);
  return(population);
}
