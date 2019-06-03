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

N <- 500
distances <- read_rds(paste0(system.file("extdata", package = "brkga"), "/MDG.21.a.n2000m200.rds"))
distances.d <- as.dist(distances)
cluster.mdp <- hclust(distances.d)
tour <- 
  tibble(
    N = 1:500,
    Cluster = cutree(cluster.mdp, 50) 
    ) %>% 
  group_by(Cluster) %>% 
  sample_n(1) %>% ungroup() %>% select(N) %>% unlist(use.names = FALSE) %>% 
  sample(50)
binaryTour <- rep(0, 500)
binaryTour[tour] <- 1
B <- doTabuSearchMI(binaryTour, distances,  alpha = 15, maxIterations =  1000)
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


#######

S <- mamdp(distanceMatrix = distances, 
           tourSize = 200, 
           populationSize = 10, 
           maxIterations = 1000, 
           maxTime = 20)
getBinaryTourFitness(S, distances)

S <- obma(distanceMatrix = distances, 
           tourSize = 200, 
           populationSize = 10, 
           maxIterations = 1000, 
           maxTime = 20)
getBinaryTourFitness(S, distances)

S <- dmamdp(distanceMatrix = distances, 
          tourSize = 200, 
          populationSize = 10, 
          lostMaxIterations = 1000, 
          maxTime = 20,
          p = .6)
getBinaryTourFitness(S, distances)
