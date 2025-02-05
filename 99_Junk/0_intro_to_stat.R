library("dplyr")
rm(list=ls());cat("\014")

a = rnorm(10000)
hist(a)

sample(1:40,50, replace = T)

sample(c("H","T"), 20, replace = T) %>% sort
sample(c("H","T"), 100, replace = T) %>% sort %>% table
sample(c("H","T"), 100, replace = T, prob = c(0.9,0.1)) %>% sort %>% table