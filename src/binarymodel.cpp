/*
 * André Leite <leite@de.ufpe.br>
 */
#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

#include "gurobi_c++.h"
#include <sstream>
#include <chrono>


//' Try to solve MDP with binary exact model
//' @param \code{distanceMatrix} Square and symmetric distance matrix 
//' @param \code{m} Tour size
//' @param \code{MAX_TIME} Its time to stop folks.
//' @param \code{THREADS} Number of threads to be used
//' @param \code{verbose} true or false
//' @return A baby 
//' @examples
//' binarymodel(...)
//' @export 
// [[Rcpp::export]]
Rcpp::List binarymodel(const arma::mat & distances, int m, double MAX_TIME = 10, int THREADS = 4, bool verbose = false) 
    {
    auto start = std::chrono::system_clock::now();
    int n = distances.n_cols; 
    if (n != distances.n_rows) // Need a Square Matrix
        Rcpp::stop("Need a square matrix.");
    if (m >= n) // Tour size senseless 
        Rcpp::stop("Must have m < n.");
    

    arma::vec binaryTour(n, arma::fill::zeros);
    Rcpp::IntegerVector tour;
    int optimstatus = 0;
    double fitness = 0;
    
    Rcpp::Rcout << "Creating Otimization Model..." << std::endl;
    GRBEnv* env = NULL;
    GRBVar* vars = NULL;
    GRBVar** prod = NULL;
    
    try {
        env = new GRBEnv();
        GRBModel model = GRBModel(*env);
        vars = new GRBVar[n];
        
        model.set(GRB_StringAttr_ModelName, "mdp");
        model.getEnv().set(GRB_IntParam_OutputFlag, verbose);
        model.getEnv().set(GRB_IntParam_Threads, THREADS);
        model.set(GRB_IntAttr_ModelSense, GRB_MAXIMIZE);
        
        // Create binary decision variables
        vars = model.addVars(n);
        for (int i = 0; i < n; ++i) 
        {
            std::ostringstream vname;
            vname << "V_" << i;
            vars[i].set(GRB_CharAttr_VType, GRB_BINARY);
            vars[i].set(GRB_StringAttr_VarName, vname.str());
        }
        
        prod = new GRBVar*[n];
        for (int i = 0; i < n; ++i) {
            prod[i] = model.addVars(n);
            for (int j = i + 1; j < n; ++j) {
                std::ostringstream vname;
                vname << "prod_" << i << "x" << j;
                prod[i][j].set(GRB_CharAttr_VType, GRB_BINARY);
                prod[i][j].set(GRB_StringAttr_VarName, vname.str());
            }
        }
        
        // Constraints (m elements in tour)
        GRBLinExpr expr = 0;
        for (int i = 0; i < n; i++) 
            expr += vars[i];
        model.addConstr(expr <= m, "TourSize");
        
        // Constraints (Linearization)
        for (int i = 0; i < n; i++) 
        {
            for (int j = i+1; j < n; j++) 
            {
                GRBLinExpr expr = prod[i][j];
                std::ostringstream vname;
                std::ostringstream vname1;
                std::ostringstream vname2;
                vname << "lin_" << i << "-" << j;
                model.addConstr(expr <= vars[j], vname.str());
                vname1 << "lin1_" << i << "-" << j;
                model.addConstr(expr <= vars[i], vname1.str());
                vname2 << "lin2_" << i << "-" << j;
                model.addConstr(expr >= vars[i]+vars[j]-1, vname2.str());
            }
        }

        // Forbid edge from node back to itself
        for (int i = 0; i < n; i++)
            prod[i][i].set(GRB_DoubleAttr_UB, 0);
        
        
        GRBLinExpr obj = 0; //Total inner tour distance;
        for (int i = 0; i < n; i++) 
        {
            for (int j = i + 1; j < n; j++) 
            {
                obj += prod[i][j]*distances(i,j);
            }
        } 
    
        model.getEnv().set(GRB_DoubleParam_TimeLimit, MAX_TIME);
        model.setObjective(obj);
        
        // Write model to file
        model.write("pdm.lp");
        
        // Optimize model
        model.optimize();
        
        optimstatus = model.get(GRB_IntAttr_Status);
        if (optimstatus == GRB_OPTIMAL || optimstatus == GRB_TIME_LIMIT) {
            fitness = model.get(GRB_DoubleAttr_ObjVal);
            Rcpp::Rcout << "Optimal objective: " << fitness << std::endl;
            // Retrieve var[n] as IntegerVector
            for(int i = 0; i < n; i++){
                double val = vars[i].get(GRB_DoubleAttr_X);
                binaryTour(i) = val;
                if (val > 0.5)
                    tour.push_back(i+1);
            }
        }
    } catch (GRBException e) {
        Rcpp::Rcout << "Error number: " << e.getErrorCode() << std::endl;
        Rcpp::Rcout << e.getMessage() << std::endl;
    } catch (...) {
        Rcpp::Rcout << "Error during optimization" << std::endl;
    }
    
    for (int i = 0; i < n; i++)
        delete[] prod[i];
    delete[] prod;
    delete[] vars;
    delete env;
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    return Rcpp::List::create(Rcpp::Named("Tour")    = tour,
                              Rcpp::Named("Binary")  = binaryTour,
                              Rcpp::Named("Fitness") = fitness,
                              Rcpp::Named("Duration")= elapsed_seconds.count(),
                              Rcpp::Named("Status")  = optimstatus      
                                  );
    }
