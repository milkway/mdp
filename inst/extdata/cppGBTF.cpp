
//' Get fitness from tour
//' @details Get fitness using the tour, m and distance matrix
//' @param \code{Tour} Set of tour's nodes.
//' @param \code{Distances} Distance matrix
//' @return A double value representing the chromosome fitness
//' @export 
// [[Rcpp::export("tour_fitness_binary")]]
double tour_fitness_binary(const arma::uvec& Tour, const arma::mat& distanceMatrix){
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
// [[Rcpp::export("tour_fitness_binary2")]]
double tour_fitness_binary2(const arma::uvec& S, const arma::mat& distances){
  double fitness = 0;
  arma::uvec idx = arma::find(S == 1);
  arma::mat tour_distances = distances.submat(idx, idx);
  unsigned n = tour_distances.n_cols;
#pragma omp parallel for reduction(+: fitness)
  for (unsigned j = 0; j < n; j++) {
    for (auto i = j + 1; i <  n; i++) {
      fitness += tour_distances(i, j);
    }
  }
  return(fitness);
} 


//' Get fitness from tour
//' @details Get fitness using the tour, m and distance matrix
//' @param \code{Tour} Set of tour's nodes.
//' @param \code{Distances} Distance matrix
//' @return A double value representing the chromosome fitness
//' @export 
// [[Rcpp::export("tour_fitness_binary4")]]
double tour_fitness_binary4(const arma::uvec& S, const arma::mat& distances){
  arma::uvec idx = arma::find(S == 1);
  arma::mat tour_distances = arma::trimatu(distances.submat(idx, idx),  1);
  return(arma::accu(tour_distances));
} 

//' Get fitness from tour
//' @details Get fitness using the tour, m and distance matrix
//' @param \code{Tour} Set of tour's nodes.
//' @param \code{Distances} Distance matrix
//' @return A double value representing the chromosome fitness
//' @export 
// [[Rcpp::export("tour_fitness_binary3")]]
double tour_fitness_binary3(const arma::uvec& S, const arma::mat& distances){
  double fitness = 0;
  arma::uvec idx = arma::find(S);
  arma::mat tour_distances(distances(idx, idx));
  unsigned n = tour_distances.n_cols;
#pragma omp parallel for reduction(+: fitness)
  for (auto i = 0; i < n; ++i) {
    auto it = tour_distances.begin_col(i) + i + 1;
    auto end = tour_distances.end_col(i);
    for (; it != end; ++it) {
      fitness += *it;
    }
  }
  return(fitness);
} 

//' Get fitness from tour
//' @details Get fitness using the tour, m and distance matrix
//' @param \code{Tour} Set of tour's nodes.
//' @param \code{Distances} Distance matrix
//' @return A double value representing the chromosome fitness
//' @export 
// [[Rcpp::export("tour_fitness_binary6")]]
double tour_fitness_binary6(const arma::uvec& S, const arma::mat& distances){
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

//' Get fitness from tour
//' @details Get fitness using the tour, m and distance matrix
//' @param \code{Tour} Set of tour's nodes.
//' @param \code{Distances} Distance matrix
//' @return A double value representing the chromosome fitness
//' @export 
// [[Rcpp::export("tour_fitness_binary7")]]
double tour_fitness_binary7(const arma::uvec& S, const arma::mat& distances){
  double fitness = 0;
  arma::uvec idx = arma::find(S);
  unsigned n = idx.size();
#pragma omp parallel for reduction(+: fitness)
  for (unsigned i = 0; i < n; ++i) {
    auto it = idx.begin() + i + 1;
    for (; it != idx.end(); ++it) {
      fitness += distances(*it, idx(i));
    }
  }
  return(fitness);
} 


//' Get fitness from tour
//' @details Get fitness using the tour, m and distance matrix
//' @param \code{Tour} Set of tour's nodes.
//' @param \code{Distances} Distance matrix
//' @return A double value representing the chromosome fitness
//' @export 
// [[Rcpp::export("tour_fitness_binary5")]]
double tour_fitness_binary5(const arma::uvec& S, const arma::mat& distances){
  return(arma::as_scalar(S.t()*distances*S)/2);
} 