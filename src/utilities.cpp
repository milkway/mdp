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
// [[Rcpp::export]]
Rcpp::List cnts(arma::uvec S,   
           double fitness, 
           const arma::mat& distances, 
           unsigned alpha =  15, 
           double rho = 1,  
           unsigned max_iterations = 1000,
           bool verbose = false) 
{
  
  double max_distance = distances.max(); // Max distance between two nodes
  //arma::uvec BestSoFar(S);
  //double fitness = getBinaryTourFitness(S, distanceMatrix);
  arma::uvec tabu_S(S); 
  double tabu_fitness = fitness;
  unsigned best_iteration = 0;
  unsigned count_iterations = 0;
  unsigned N = distances.n_cols; // Number of nodes
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
    //unsigned max_rand = max_indices(arma::randi(arma::distr_param(0,max_indices.size()-1)));
    for (unsigned i = 0; i < max_indices.size(); i++){
      unsigned u = CN_X(max_indices(i) % CN_X.size());
      unsigned v = CN_Y(max_indices(i) / CN_X.size());
      // Swap u and v if admissible
      if (((Tenure(u) == 0) && (Tenure(v) == 0)) || 
          (fitness < (tabu_fitness + max_delta))){
        tabu_S(u) = 0;
        tabu_S(v) = 1;
        tabu_fitness += max_delta;
        gain +=  distances.col(v) - distances.col(u);
        unsigned index = (unsigned)std::floor((((double)(count_iterations%1500))/100.0));
        Tenure(u) = alpha*alpha_multiplier(index);
        Tenure(v) = unsigned(0.7*Tenure(u));
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
// [[Rcpp::export]]
Rcpp::List cnts_sugar(arma::uvec S,   
                const arma::mat& distances, 
                unsigned alpha =  15, 
                double rho = 1,  
                unsigned max_iterations = 1000,
                bool verbose = false) 
{
  
  double max_distance = distances.max(); // Max distance between two nodes
  double fitness = tour_fitness_binary(S, distances);
  arma::uvec tabu_S(S); 
  double tabu_fitness = fitness;
  unsigned best_iteration = 0;
  unsigned count_iterations = 0;
  unsigned N = distances.n_cols; // Number of nodes
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
      unsigned u = CN_X(idxs(0));
      unsigned v = CN_Y(idxs(1));
      // Swap u and v if admissible
      if (((Tenure(u) == 0) && (Tenure(v) == 0)) || 
          (fitness < (tabu_fitness + max_delta))){
        tabu_S(u) = 0;
        tabu_S(v) = 1;
        tabu_fitness += max_delta;
        gain +=  distances.col(v) - distances.col(u);
        unsigned index = (unsigned)std::floor((((double)(count_iterations%1500))/100.0));
        Tenure(u) = alpha*alpha_multiplier(index);
        Tenure(v) = unsigned(0.7*Tenure(u));
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
// [[Rcpp::export]]
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


//' The crossover operator of the MAMDP algorithm 
//' Cross Over between S_a and S_b
//' @param \code{S_a} Parent A
//' @param \code{S_b} Parent B
//' @param \code{distances} Square and symmetric distance matrix 
//' @return A baby 
//' @examples
//' crossover()
// [[Rcpp::export]]
arma::uvec crossover(const arma::uvec& S_a, const arma::uvec& S_b, const arma::mat& distances) 
{
  arma::uvec child(S_a % S_b); // Backbone
  arma::uvec poolA = arma::find((S_a + child) == 1); // The other poor little guys in S_a
  arma::uvec poolB = arma::find((S_b + child) == 1); // The other poor little guys in S_b
  unsigned m = arma::sum(S_a);
  unsigned order = 0;
  unsigned goingIn = 0;
  while(arma::sum(child) < m){
    arma::mat deltaA = distances(arma::find(child == 1), poolA);
    order = arma::index_max(arma::sum(deltaA, 0));
    goingIn = poolA(order);
    child(goingIn) = 1;
    poolA.shed_row(order);
    if (arma::sum(child) >= m) break;
    arma::mat deltaB = distances(arma::find(child == 1), poolB);
    order = arma::index_max(arma::sum(deltaB, 0));
    goingIn = poolB(order);
    child(goingIn) = 1;
    poolB.shed_row(order); 
  }
  //Rcpp::Rcout << "S0sum "<< arma::sum(child) << std::endl;
  return(child);
}

//' The crossover operator of the MAMDP algorithm 
//' Cross Over between S_a and S_b
//' @param \code{S_a} Parent A
//' @param \code{S_b} Parent B
//' @param \code{distances} Square and symmetric distance matrix 
//' @return A baby 
//' @examples
//' crossover()
// [[Rcpp::export]]
arma::uvec crossover_rand(const arma::uvec& S_a, const arma::uvec& S_b, const arma::mat& distances) 
{
  arma::uvec child(S_a % S_b); // Backbone
  arma::uvec poolA = arma::find((S_a + child) == 1); // The other poor little guys in S_a
  arma::uvec poolB = arma::find((S_b + child) == 1); // The other poor little guys in S_b
  unsigned m = arma::sum(S_a);
  unsigned order = 0;
  unsigned goingIn = 0;
  while(arma::sum(child) < m){
    arma::mat deltaA = distances(arma::find(child == 1), poolA);
    order = arma::index_max(arma::sum(deltaA, 0));
    goingIn = poolA(order);
    child(goingIn) = 1;
    arma::uvec gringos =  arma::find(child + S_a + S_b == 0);
    poolA(order) = gringos(arma::randi(arma::distr_param(0,gringos.size()-1)));
    if (arma::sum(child) >= m) break;
    arma::mat deltaB = distances(arma::find(child == 1), poolB);
    order = arma::index_max(arma::sum(deltaB, 0));
    goingIn = poolB(order);
    child(goingIn) = 1;
    gringos =  arma::find(child + S_a + S_b == 0);
    poolB(order) = gringos(arma::randi(arma::distr_param(0,gringos.size()-1)));
  }
  //Rcpp::Rcout << "S0sum "<< arma::sum(child) << std::endl;
  return(child);
}

// Eq. 8 of http://dx.doi.org/10.1016/j.ejor.2013.06.002
// Scale Function used in population update
arma::vec normalized_function(arma::vec input){
  double max_in = arma::max(input);
  double min_in = arma::min(input);
  return((input - min_in)/(max_in - min_in + 1));
}

//' Def. 1  of http://dx.doi.org/10.1016/j.ejor.2013.06.002
//' Distance between to individuals;
//' @details Distance between to individuals: m - sum(S0XS1);
//' @param \code{S0} Individual.
//' @param \code{S1} Individual.
//' @return A int, number of moves to go from one individual to the other.
//' @export 
// [[Rcpp::export]]
double solutions_distance(const arma::uvec& S0, const arma::uvec& S1){
  unsigned m = arma::sum(S0);
  if (m != arma::sum(S1))
    Rcpp::stop("Tour sizes must be iguals");
  return(m - arma::sum(S0 % S1));
}

//' Def. 2  of http://dx.doi.org/10.1016/j.ejor.2013.06.002
//' Distance between a individual and a population;
//' @details Distance between to individuals in population: min(m - sum(S0XS), S in Population);
//' @param \code{S0} Individual.
//' @param \code{Population} Target Population.
//' @return A int, min distance between a indivitual and a population.
//' @export 
// [[Rcpp::export]]
double solution_to_population_distance(const arma::uvec& S0, const arma::umat& population, unsigned tour_size){
  unsigned m = tour_size;
  if (arma::sum(population.col(0)) == 0) 
    return(m);
  auto M(population);
  if (any(arma::sum(M,0) - m != 0 ))
    Rcpp::stop("Tour sizes must be iguals");
  M.each_col() %= S0; // Multiply each individual by S0, element wise.  
  return((double)arma::min(m - arma::sum(M,0)));
}

//' Def. 2  of http://dx.doi.org/10.1016/j.ejor.2013.06.002
//' Distance between a individual and a population;
//' Distance between a individual and a population by index;
//' @details Distance between to individuals in population: min(m - sum(S0XS), S in Population);
//' @param \code{S0} Individual.
//' @param \code{Population} Target Population.
//' @return A int, min distance between a indivitual and a population.
//' @export 
// [[Rcpp::export]]
double index_to_population_distance(const unsigned idx, const arma::umat& population){
  unsigned m = arma::sum(population.col(0));
  if (!((idx >= 0)&&(idx < population.n_cols)))
    Rcpp::stop("Sorry. Here we count from 0 to N-1");
  auto M(population);
  arma::uvec S0 = M.col(idx);
  M.shed_col(idx);
  M.each_col() %= S0; // Multiply each individual by S0, element wise.  
  return((double)arma::min(m - arma::sum(M,0)));
}

//' Tabu search in a random solution
//' @details generate a random solution, apply tabu search and return the new solution and its fitness
//' @param \code{distances} Distance matrix
//' @param \code{tour_size} You know!
//' @param \code{alpha} You know!
//' @param \code{rho} You know!
//' @param \code{max_iteration} You know!
//' @return List with solution and its fitness
//' @export
// [[Rcpp::export]]
Rcpp::List tabu_solution_rand(const arma::mat& distances, unsigned tour_size, unsigned alpha = 15, double rho = 1, unsigned max_iterations = 50000){
  unsigned N = distances.n_cols;
  arma::uvec nodes = arma::shuffle(arma::linspace<arma::uvec>(0, N-1, N));
  arma::uvec candidate(N, arma::fill::zeros);
  candidate.elem(nodes.subvec(0, tour_size - 1)) += 1;
  double fitness = tour_fitness_binary(candidate, distances);
  Rcpp::List tabu = cnts(candidate, fitness, distances, alpha, rho, max_iterations, false);
  return Rcpp::List::create(Rcpp::Named("S")       = candidate,
                            Rcpp::Named("fitness") = fitness);
}

//' MAMDP population update
//' Algorithm 4. Update population strategy of the MAMDP algorithm
//' Ref: http://dx.doi.org/10.1016/j.ejor.2013.06.002
//' @param \code{S} population candidate
//' @param \code{Population} population to be updated
//' @return New population 
//' @examples
//' updatePopulation()
// [[Rcpp::export]]
Rcpp::List update_population_mamdp(arma::uvec S, arma::umat& population, arma::vec& fitness,  const arma::mat& distances, double beta = 0.6){
  unsigned N = S.size();
  unsigned P = population.n_cols;
  population.insert_cols(P, S);
  fitness.resize(P+1);
  fitness(P) = tour_fitness_binary(S, distances);
  arma::vec distance(population.n_cols, arma::fill::zeros);
  for(unsigned i = 0; i < population.n_cols; i++){
    distance(i) = index_to_population_distance(i, population);
  }
  arma::vec score = beta*(normalized_function(fitness)) + (1-beta)*(normalized_function(distance));
  population.shed_col(arma::index_min(score));
  fitness.shed_row(arma::index_min(score));
  return Rcpp::List::create(Rcpp::Named("population") = population,
                            Rcpp::Named("fitness")    = fitness);
}


//' Pool initialization for MAMDP
//' Section 2.2 of http://dx.doi.org/10.1016/j.ejor.2013.06.002
//' @details Get the initial populatio
//' @param \code{tourSize} Ok. Stop it.
//' @param \code{Distances} Distance matrix
//' @param \code{populationSize} You know!
//' @return A matrix where each column is a individual in the population
//' @export
// [[Rcpp::export]]
Rcpp::List initialize_population_mamdp(const arma::mat& distances,
                                 unsigned tour_size,
                                 unsigned population_size = 10,
                                 unsigned tabu_max_iterations = 50000,
                                 unsigned tabu_multiplier = 3,
                                 double tabu_rho = 1,
                                 unsigned tabu_alpha = 15){
  unsigned N = distances.n_cols;
  unsigned count = 1;
  arma::umat population(N, population_size*tabu_multiplier, arma::fill::zeros);
  arma::vec population_fitness(population_size*tabu_multiplier, arma::fill::zeros);
  Rcpp::List tabu_candidate = tabu_solution_rand(distances, tour_size, tabu_alpha, tabu_rho, tabu_max_iterations);
  population.col(0) = Rcpp::as<arma::uvec>(tabu_candidate["S"]);
  population_fitness(0) = tabu_candidate["fitness"];
  while(count < population_size*tabu_multiplier){
      tabu_candidate = tabu_solution_rand(distances, tour_size, tabu_alpha, tabu_rho, tabu_max_iterations);
      if (solution_to_population_distance(tabu_candidate["S"], population.cols(0, count-1), tour_size) > 0.5){
        population.col(count) = Rcpp::as<arma::uvec>(tabu_candidate["S"]);
        population_fitness(count) = tabu_candidate["fitness"];
        count++;
      }
      Rcpp::checkUserInterrupt();
    }

  arma::uvec idx_sorted = arma::sort_index(population_fitness, "descend");
  arma::uvec idx_best = idx_sorted.head(population_size);
  arma::umat best_pop = population.cols(idx_best);
  arma::vec best_fitness = population_fitness.elem(idx_best);
  return Rcpp::List::create(Rcpp::Named("population") = best_pop,
                            Rcpp::Named("fitness")    = best_fitness);
}



