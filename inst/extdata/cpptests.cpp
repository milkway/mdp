#include <RcppArmadillo.h>
using namespace Rcpp;
//[[Rcpp::depends("RcppArmadillo")]]

// This is a simple example of exporting a C++ function to R. You can
// source this function into an R session using the Rcpp::sourceCpp 
// function (or via the Source button on the editor toolbar). Learn
// more about Rcpp at:
//
//   http://www.rcpp.org/
//   http://adv-r.had.co.nz/Rcpp.html
//   http://gallery.rcpp.org/
//

// [[Rcpp::export]]
NumericVector timesTwo(NumericVector x) {
  return x * 2;
}

// [[Rcpp::export]]
arma::vec testI2S(arma::mat M){
  arma::uvec indices = arma::find(M > 0.5);
  arma::umat t       = arma::ind2sub( arma::size(M), indices );
  arma::vec v = t.each_col();    
  return(v);
}


// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically 
// run after the compilation.
//

/*** R
timesTwo(42)
*/
