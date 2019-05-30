// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// binarymodel
Rcpp::List binarymodel(const arma::mat& distances, int m, double MAX_TIME, int THREADS, bool verbose);
RcppExport SEXP _mdp_binarymodel(SEXP distancesSEXP, SEXP mSEXP, SEXP MAX_TIMESEXP, SEXP THREADSSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type distances(distancesSEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    Rcpp::traits::input_parameter< double >::type MAX_TIME(MAX_TIMESEXP);
    Rcpp::traits::input_parameter< int >::type THREADS(THREADSSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(binarymodel(distances, m, MAX_TIME, THREADS, verbose));
    return rcpp_result_gen;
END_RCPP
}
// findBestPopFitness
int findBestPopFitness(const arma::umat& Population, const arma::mat& distanceMatrix);
RcppExport SEXP _mdp_findBestPopFitness(SEXP PopulationSEXP, SEXP distanceMatrixSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::umat& >::type Population(PopulationSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type distanceMatrix(distanceMatrixSEXP);
    rcpp_result_gen = Rcpp::wrap(findBestPopFitness(Population, distanceMatrix));
    return rcpp_result_gen;
END_RCPP
}
// doCrossOver
arma::uvec doCrossOver(const arma::uvec& S_a, const arma::uvec& S_b, const arma::mat& distanceMatrix);
RcppExport SEXP _mdp_doCrossOver(SEXP S_aSEXP, SEXP S_bSEXP, SEXP distanceMatrixSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::uvec& >::type S_a(S_aSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type S_b(S_bSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type distanceMatrix(distanceMatrixSEXP);
    rcpp_result_gen = Rcpp::wrap(doCrossOver(S_a, S_b, distanceMatrix));
    return rcpp_result_gen;
END_RCPP
}
// doBackboneCrossOver
arma::umat doBackboneCrossOver(const arma::uvec& S_a, const arma::uvec& S_b, const arma::mat& distanceMatrix);
RcppExport SEXP _mdp_doBackboneCrossOver(SEXP S_aSEXP, SEXP S_bSEXP, SEXP distanceMatrixSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::uvec& >::type S_a(S_aSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type S_b(S_bSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type distanceMatrix(distanceMatrixSEXP);
    rcpp_result_gen = Rcpp::wrap(doBackboneCrossOver(S_a, S_b, distanceMatrix));
    return rcpp_result_gen;
END_RCPP
}
// doTabuSearchMI
arma::uvec doTabuSearchMI(arma::uvec S, const arma::mat& distanceMatrix, int alpha, double rhoOver2, int maxIterations);
RcppExport SEXP _mdp_doTabuSearchMI(SEXP SSEXP, SEXP distanceMatrixSEXP, SEXP alphaSEXP, SEXP rhoOver2SEXP, SEXP maxIterationsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::uvec >::type S(SSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type distanceMatrix(distanceMatrixSEXP);
    Rcpp::traits::input_parameter< int >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double >::type rhoOver2(rhoOver2SEXP);
    Rcpp::traits::input_parameter< int >::type maxIterations(maxIterationsSEXP);
    rcpp_result_gen = Rcpp::wrap(doTabuSearchMI(S, distanceMatrix, alpha, rhoOver2, maxIterations));
    return rcpp_result_gen;
END_RCPP
}
// doTabuSearchML
arma::uvec doTabuSearchML(arma::uvec S, const arma::mat& distanceMatrix, int alpha, double rhoOver2, int lostMaxIterations);
RcppExport SEXP _mdp_doTabuSearchML(SEXP SSEXP, SEXP distanceMatrixSEXP, SEXP alphaSEXP, SEXP rhoOver2SEXP, SEXP lostMaxIterationsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::uvec >::type S(SSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type distanceMatrix(distanceMatrixSEXP);
    Rcpp::traits::input_parameter< int >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double >::type rhoOver2(rhoOver2SEXP);
    Rcpp::traits::input_parameter< int >::type lostMaxIterations(lostMaxIterationsSEXP);
    rcpp_result_gen = Rcpp::wrap(doTabuSearchML(S, distanceMatrix, alpha, rhoOver2, lostMaxIterations));
    return rcpp_result_gen;
END_RCPP
}
// updatePopulation
arma::umat updatePopulation(arma::uvec S, arma::umat Population, const arma::mat& distanceMatrix, double beta);
RcppExport SEXP _mdp_updatePopulation(SEXP SSEXP, SEXP PopulationSEXP, SEXP distanceMatrixSEXP, SEXP betaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::uvec >::type S(SSEXP);
    Rcpp::traits::input_parameter< arma::umat >::type Population(PopulationSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type distanceMatrix(distanceMatrixSEXP);
    Rcpp::traits::input_parameter< double >::type beta(betaSEXP);
    rcpp_result_gen = Rcpp::wrap(updatePopulation(S, Population, distanceMatrix, beta));
    return rcpp_result_gen;
END_RCPP
}
// updatePopulationByRank
arma::umat updatePopulationByRank(arma::uvec S, arma::umat Population, const arma::mat& distanceMatrix, double beta);
RcppExport SEXP _mdp_updatePopulationByRank(SEXP SSEXP, SEXP PopulationSEXP, SEXP distanceMatrixSEXP, SEXP betaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::uvec >::type S(SSEXP);
    Rcpp::traits::input_parameter< arma::umat >::type Population(PopulationSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type distanceMatrix(distanceMatrixSEXP);
    Rcpp::traits::input_parameter< double >::type beta(betaSEXP);
    rcpp_result_gen = Rcpp::wrap(updatePopulationByRank(S, Population, distanceMatrix, beta));
    return rcpp_result_gen;
END_RCPP
}
// initializeOBP
arma::umat initializeOBP(const arma::mat& distanceMatrix, int tourSize, int populationSize, int maxIterations, int multiplier);
RcppExport SEXP _mdp_initializeOBP(SEXP distanceMatrixSEXP, SEXP tourSizeSEXP, SEXP populationSizeSEXP, SEXP maxIterationsSEXP, SEXP multiplierSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type distanceMatrix(distanceMatrixSEXP);
    Rcpp::traits::input_parameter< int >::type tourSize(tourSizeSEXP);
    Rcpp::traits::input_parameter< int >::type populationSize(populationSizeSEXP);
    Rcpp::traits::input_parameter< int >::type maxIterations(maxIterationsSEXP);
    Rcpp::traits::input_parameter< int >::type multiplier(multiplierSEXP);
    rcpp_result_gen = Rcpp::wrap(initializeOBP(distanceMatrix, tourSize, populationSize, maxIterations, multiplier));
    return rcpp_result_gen;
END_RCPP
}
// initializePool
arma::umat initializePool(const arma::mat& distanceMatrix, int tourSize, int populationSize, int maxIterations, int multiplier);
RcppExport SEXP _mdp_initializePool(SEXP distanceMatrixSEXP, SEXP tourSizeSEXP, SEXP populationSizeSEXP, SEXP maxIterationsSEXP, SEXP multiplierSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type distanceMatrix(distanceMatrixSEXP);
    Rcpp::traits::input_parameter< int >::type tourSize(tourSizeSEXP);
    Rcpp::traits::input_parameter< int >::type populationSize(populationSizeSEXP);
    Rcpp::traits::input_parameter< int >::type maxIterations(maxIterationsSEXP);
    Rcpp::traits::input_parameter< int >::type multiplier(multiplierSEXP);
    rcpp_result_gen = Rcpp::wrap(initializePool(distanceMatrix, tourSize, populationSize, maxIterations, multiplier));
    return rcpp_result_gen;
END_RCPP
}
// initializePoolML
arma::umat initializePoolML(const arma::mat& distanceMatrix, int tourSize, int populationSize, int lostMaxIterations, int multiplier);
RcppExport SEXP _mdp_initializePoolML(SEXP distanceMatrixSEXP, SEXP tourSizeSEXP, SEXP populationSizeSEXP, SEXP lostMaxIterationsSEXP, SEXP multiplierSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type distanceMatrix(distanceMatrixSEXP);
    Rcpp::traits::input_parameter< int >::type tourSize(tourSizeSEXP);
    Rcpp::traits::input_parameter< int >::type populationSize(populationSizeSEXP);
    Rcpp::traits::input_parameter< int >::type lostMaxIterations(lostMaxIterationsSEXP);
    Rcpp::traits::input_parameter< int >::type multiplier(multiplierSEXP);
    rcpp_result_gen = Rcpp::wrap(initializePoolML(distanceMatrix, tourSize, populationSize, lostMaxIterations, multiplier));
    return rcpp_result_gen;
END_RCPP
}
// getBinaryTourFitness
double getBinaryTourFitness(const arma::uvec& Tour, const arma::mat& distanceMatrix);
RcppExport SEXP _mdp_getBinaryTourFitness(SEXP TourSEXP, SEXP distanceMatrixSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::uvec& >::type Tour(TourSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type distanceMatrix(distanceMatrixSEXP);
    rcpp_result_gen = Rcpp::wrap(getBinaryTourFitness(Tour, distanceMatrix));
    return rcpp_result_gen;
END_RCPP
}
// getTourFitness
double getTourFitness(const arma::uvec& Tour, const arma::mat& distanceMatrix);
RcppExport SEXP _mdp_getTourFitness(SEXP TourSEXP, SEXP distanceMatrixSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::uvec& >::type Tour(TourSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type distanceMatrix(distanceMatrixSEXP);
    rcpp_result_gen = Rcpp::wrap(getTourFitness(Tour, distanceMatrix));
    return rcpp_result_gen;
END_RCPP
}
// getSolutionsDistance
int getSolutionsDistance(const arma::uvec& S0, const arma::uvec& S1);
RcppExport SEXP _mdp_getSolutionsDistance(SEXP S0SEXP, SEXP S1SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::uvec& >::type S0(S0SEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type S1(S1SEXP);
    rcpp_result_gen = Rcpp::wrap(getSolutionsDistance(S0, S1));
    return rcpp_result_gen;
END_RCPP
}
// getSolutionToPopulationDistance
int getSolutionToPopulationDistance(const arma::uvec& S0, const arma::umat& Population);
RcppExport SEXP _mdp_getSolutionToPopulationDistance(SEXP S0SEXP, SEXP PopulationSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::uvec& >::type S0(S0SEXP);
    Rcpp::traits::input_parameter< const arma::umat& >::type Population(PopulationSEXP);
    rcpp_result_gen = Rcpp::wrap(getSolutionToPopulationDistance(S0, Population));
    return rcpp_result_gen;
END_RCPP
}
// getAverageDistanceToPopulation
int getAverageDistanceToPopulation(const arma::uvec& S0, const arma::umat& Population);
RcppExport SEXP _mdp_getAverageDistanceToPopulation(SEXP S0SEXP, SEXP PopulationSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::uvec& >::type S0(S0SEXP);
    Rcpp::traits::input_parameter< const arma::umat& >::type Population(PopulationSEXP);
    rcpp_result_gen = Rcpp::wrap(getAverageDistanceToPopulation(S0, Population));
    return rcpp_result_gen;
END_RCPP
}
// getAverageDistanceToPopulationByIndex
int getAverageDistanceToPopulationByIndex(int index, const arma::umat& Population);
RcppExport SEXP _mdp_getAverageDistanceToPopulationByIndex(SEXP indexSEXP, SEXP PopulationSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type index(indexSEXP);
    Rcpp::traits::input_parameter< const arma::umat& >::type Population(PopulationSEXP);
    rcpp_result_gen = Rcpp::wrap(getAverageDistanceToPopulationByIndex(index, Population));
    return rcpp_result_gen;
END_RCPP
}
// getSolutionToPopulationDistanceByIndex
int getSolutionToPopulationDistanceByIndex(int index, const arma::umat& Population);
RcppExport SEXP _mdp_getSolutionToPopulationDistanceByIndex(SEXP indexSEXP, SEXP PopulationSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type index(indexSEXP);
    Rcpp::traits::input_parameter< const arma::umat& >::type Population(PopulationSEXP);
    rcpp_result_gen = Rcpp::wrap(getSolutionToPopulationDistanceByIndex(index, Population));
    return rcpp_result_gen;
END_RCPP
}
// mamdp
arma::uvec mamdp(const arma::mat& distanceMatrix, int tourSize, int populationSize, int maxIterations, double maxTime);
RcppExport SEXP _mdp_mamdp(SEXP distanceMatrixSEXP, SEXP tourSizeSEXP, SEXP populationSizeSEXP, SEXP maxIterationsSEXP, SEXP maxTimeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type distanceMatrix(distanceMatrixSEXP);
    Rcpp::traits::input_parameter< int >::type tourSize(tourSizeSEXP);
    Rcpp::traits::input_parameter< int >::type populationSize(populationSizeSEXP);
    Rcpp::traits::input_parameter< int >::type maxIterations(maxIterationsSEXP);
    Rcpp::traits::input_parameter< double >::type maxTime(maxTimeSEXP);
    rcpp_result_gen = Rcpp::wrap(mamdp(distanceMatrix, tourSize, populationSize, maxIterations, maxTime));
    return rcpp_result_gen;
END_RCPP
}
// obma
arma::uvec obma(const arma::mat& distanceMatrix, int tourSize, int populationSize, int maxIterations, double maxTime);
RcppExport SEXP _mdp_obma(SEXP distanceMatrixSEXP, SEXP tourSizeSEXP, SEXP populationSizeSEXP, SEXP maxIterationsSEXP, SEXP maxTimeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type distanceMatrix(distanceMatrixSEXP);
    Rcpp::traits::input_parameter< int >::type tourSize(tourSizeSEXP);
    Rcpp::traits::input_parameter< int >::type populationSize(populationSizeSEXP);
    Rcpp::traits::input_parameter< int >::type maxIterations(maxIterationsSEXP);
    Rcpp::traits::input_parameter< double >::type maxTime(maxTimeSEXP);
    rcpp_result_gen = Rcpp::wrap(obma(distanceMatrix, tourSize, populationSize, maxIterations, maxTime));
    return rcpp_result_gen;
END_RCPP
}
// dmamdp
arma::uvec dmamdp(const arma::mat& distanceMatrix, int tourSize, int populationSize, double maxTime, int lostMaxIterations, double p);
RcppExport SEXP _mdp_dmamdp(SEXP distanceMatrixSEXP, SEXP tourSizeSEXP, SEXP populationSizeSEXP, SEXP maxTimeSEXP, SEXP lostMaxIterationsSEXP, SEXP pSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type distanceMatrix(distanceMatrixSEXP);
    Rcpp::traits::input_parameter< int >::type tourSize(tourSizeSEXP);
    Rcpp::traits::input_parameter< int >::type populationSize(populationSizeSEXP);
    Rcpp::traits::input_parameter< double >::type maxTime(maxTimeSEXP);
    Rcpp::traits::input_parameter< int >::type lostMaxIterations(lostMaxIterationsSEXP);
    Rcpp::traits::input_parameter< double >::type p(pSEXP);
    rcpp_result_gen = Rcpp::wrap(dmamdp(distanceMatrix, tourSize, populationSize, maxTime, lostMaxIterations, p));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_mdp_binarymodel", (DL_FUNC) &_mdp_binarymodel, 5},
    {"_mdp_findBestPopFitness", (DL_FUNC) &_mdp_findBestPopFitness, 2},
    {"_mdp_doCrossOver", (DL_FUNC) &_mdp_doCrossOver, 3},
    {"_mdp_doBackboneCrossOver", (DL_FUNC) &_mdp_doBackboneCrossOver, 3},
    {"_mdp_doTabuSearchMI", (DL_FUNC) &_mdp_doTabuSearchMI, 5},
    {"_mdp_doTabuSearchML", (DL_FUNC) &_mdp_doTabuSearchML, 5},
    {"_mdp_updatePopulation", (DL_FUNC) &_mdp_updatePopulation, 4},
    {"_mdp_updatePopulationByRank", (DL_FUNC) &_mdp_updatePopulationByRank, 4},
    {"_mdp_initializeOBP", (DL_FUNC) &_mdp_initializeOBP, 5},
    {"_mdp_initializePool", (DL_FUNC) &_mdp_initializePool, 5},
    {"_mdp_initializePoolML", (DL_FUNC) &_mdp_initializePoolML, 5},
    {"_mdp_getBinaryTourFitness", (DL_FUNC) &_mdp_getBinaryTourFitness, 2},
    {"_mdp_getTourFitness", (DL_FUNC) &_mdp_getTourFitness, 2},
    {"_mdp_getSolutionsDistance", (DL_FUNC) &_mdp_getSolutionsDistance, 2},
    {"_mdp_getSolutionToPopulationDistance", (DL_FUNC) &_mdp_getSolutionToPopulationDistance, 2},
    {"_mdp_getAverageDistanceToPopulation", (DL_FUNC) &_mdp_getAverageDistanceToPopulation, 2},
    {"_mdp_getAverageDistanceToPopulationByIndex", (DL_FUNC) &_mdp_getAverageDistanceToPopulationByIndex, 2},
    {"_mdp_getSolutionToPopulationDistanceByIndex", (DL_FUNC) &_mdp_getSolutionToPopulationDistanceByIndex, 2},
    {"_mdp_mamdp", (DL_FUNC) &_mdp_mamdp, 5},
    {"_mdp_obma", (DL_FUNC) &_mdp_obma, 5},
    {"_mdp_dmamdp", (DL_FUNC) &_mdp_dmamdp, 6},
    {NULL, NULL, 0}
};

RcppExport void R_init_mdp(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
