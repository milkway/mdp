library(tidyverse)


distances <- c(0, 1, 1, 1, 1, 4, 2, 1,
               1, 0, 3, 3, 3, 2, 1, 1, 
               1, 3, 0, 4, 2, 2, 1, 2,
               1, 3, 4, 0, 3, 3, 2, 1,
               1, 3, 2, 3, 0, 1, 2, 3,
               4, 2, 2, 3, 1, 0, 3, 1,
               2, 1, 1, 2, 2, 3, 0, 1,
               1, 1, 2, 1, 3, 1, 1, 0) %>% matrix(nrow = 8, byrow = TRUE)

colnames(distances) <- 1:8
rownames(distances) <- 1:8

distances <- read_rds(paste0(system.file("extdata", package = "brkga"), "/MDG.1.a.n500m50.rds"))

rst <- binarymodel(distances, m = 50, MAX_TIME = 600, THREADS = 8,  verbose = TRUE)

N <- 2000
distances <- read_rds(paste0(system.file("extdata", package = "brkga"), "/MDG.21.b.n2000m200.rds"))
distances.d <- as.dist(distances)
cluster.mdp <- hclust(distances.d)
tour <- 
  tibble(
    N = 1:2000,
    Cluster = cutree(cluster.mdp, 200) 
    ) %>% 
  group_by(Cluster) %>% 
  sample_n(1) %>% ungroup() %>% select(N) %>% unlist(use.names = FALSE) %>% 
  sample(200)

binaryTour <- rep(0, 2000)
binaryTour[tour] <- 1
B <- doTabuSearchMI(binaryTour, distances,  alpha = 15, maxIterations =  10)
getBinaryTourFitness(B, distances)

binaryTour <- rep(0, 2000)
binaryTour[sample(1:2000,200)] <- 1
B <- doTabuSearchML(binaryTour, distances,  alpha = 15, lostMaxIterations =  10000, rhoOver2 = 2)
getBinaryTourFitness(B, distances)

B <- doTabuSearchParallel(binaryTour, distances,  alpha = 15, maxIterations =  50000, rhoOver2 = 2)
fitness <- getBinaryTourFitness(binaryTour, distances)

rst1 <- cnts(binaryTour,  fitness, distances,  alpha = 15, max_iterations =  50000, rho = 1, verbose = TRUE)

rst2 <- cnts_sugar(binaryTour,  distances,  alpha = 15, max_iterations =  50000, rho = 1, verbose = TRUE)

test(binaryTour, fitness, distances)

getBinaryTourFitness(rst$S, distances)

#microbenchmark::microbenchmark(F = doTabuSearch(binaryTour, distances, rhoOver2 = 1, alpha = 15, maxIterations =  1000), times = 10)
getBinaryTourFitness(B, distances)


P <- initializeOPB(distances, 50, 10, 1000, 2)
apply(P, 2, function(x){getBinaryTourFitness(x, distances)})

P <- initializePool(distances, 50, 10, 100, 2)
apply(P, 2, function(x){getBinaryTourFitness(x, distances)})
C = doBackboneCrossOver(P[,1], P[,8], distances)
getBinaryTourFitness(P[,9], distances)
P2 <- updatePopulationByRank(S = C[,1], Population = P, distanceMatrix = distances, beta = .6) 
P2 <- updatePopulationMAMDP(S = C[,1], Population = P, distanceMatrix = distances, beta = .6) 
apply(P2, 2, function(x){getBinaryTourFitness(x, distances)})

###################33

S <- rep(0, 2000)
S[sample(1:2000,200)] <- 1
.5*(t(S)%*%distances%*%S)
sum(diag(distances))
microbenchmark::microbenchmark(
  gB = getBinaryTourFitness(S, distances),
  b1 = tour_fitness_binary(S, distances),
  b2 = tour_fitness_binary2(S, distances),
  b3 = tour_fitness_binary3(S, distances),
  b4 = tour_fitness_binary4(S, distances),
  b5 = tour_fitness_binary5(S, distances),
  b6 = tour_fitness_binary6(S, distances),
  b7 = tour_fitness_binary7(S, distances),
  times = 100
)

#######

S <- mamdp(distanceMatrix = distances, 
           tourSize = 200, 
           populationSize = 10, 
           maxIterations = 10000, 
           maxTime = 40, 
           rhoOver2 = 1, 
           multiplier = 1, 
           verbose = TRUE)
S$data

getBinaryTourFitness(S$tour, distances)

S <- obma(distanceMatrix = distances, 
          tourSize = 200, 
          populationSize = 10, 
          maxIterations = 500,  
          maxTime = 60,
          multiplier = 3, 
          rhoOver2 = 2, 
          verbose = TRUE)
S$data
getBinaryTourFitness(S, distances)

S <- dmamdp(distanceMatrix = distances, 
          tourSize = 200, 
          populationSize = 10, 
          lostMaxIterations = 10000, 
          maxTime = 60, 
          multiplier = 3, 
          rhoOver2 = 2, 
          verbose = TRUE,
          p = .6)
S$data
getBinaryTourFitness(S, distances)

###

rst1 <- cnts(S, distances,  alpha = 15, max_iterations =  50000, rho = 1, verbose = TRUE)
rst1$fitness
tour_fitness_binary(rst1$S, distances)

S <- rep(0, 2000)
S[sample(1:2000,200)] <- 1
rst2 <- cnts_sugar(S,  distances,  alpha = 15, max_iterations =  50000, rho = 2, verbose = TRUE)
rst2$fitness
rst2$iterations
tour_fitness_binary(rst2$S, distances)

S <- rep(0, 2000)
S[sample(1:2000,200)] <- 1
fitness <- tour_fitness_binary(S, distances)
microbenchmark::microbenchmark(
  F1 = cnts(S,  fitness, distances,  alpha = 15, max_iterations =  50000, rho = 1, verbose = FALSE), 
  F2 = cnts_sugar(S,  distances,  alpha = 15, max_iterations =  50000, rho = 1, verbose = FALSE),
  times = 10
)
