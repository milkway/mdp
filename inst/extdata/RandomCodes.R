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

distances <- read_rds(paste0(system.file("extdata", package = "brkga"), "/MDG.1.a.n500m50.rds"))
distances.d <- as.dist(distances)
cluster.mdp <- hclust(distances.d, method = "ward.D2")
plot(cluster.mdp, hang = -1)
rect.hclust(cluster.mdp,50)



cluster.mdp <- hclust(distances.d)
tour <- 
  tibble(
    N = 1:500,
    Cluster = cutree(cluster.mdp, 50) 
    ) %>% 
  group_by(Cluster) %>% 
  sample_n(1) %>% ungroup() %>% select(N) %>% unlist(use.names = FALSE) %>% 
  sample(50)
sum(distances[tour,tour])/2

binaryTour <- rep(0, 500)
binaryTour[tour] <- 1
system.time(B <- doTabuSearch(binaryTour, distances, alpha = 15, maxIterations =  5000))
getBinaryTourFitness(B, distances)