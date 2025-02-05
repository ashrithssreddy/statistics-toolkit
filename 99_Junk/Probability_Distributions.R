dev.off()
library("dplyr")
library("ggplot2")
rm(list=ls()); cat("\014")

#### uniform distribution ####
set.seed(2022)
a = 1; b = 5
x = runif(1000, min=a, max=b)
(b + a) / 2 # mean
(b - a)^2 / 12 # var

x %>% summary
x %>% sd
x %>% var

# density and density curve
dunif(x, min=a, max=b) 
curve(dunif(x, min=a, max=b), from=a, to=b, main="PDF of Uniform Distribution", ylab="Density", xlab="x", col="blue")

# histogram
ggplot() + 
  aes(x = x) + 
  geom_histogram()

hist(x, freq = F)
plot(density(x))

#### normal distribution ####
set.seed(2022)
x = rnorm(1000)
x %>% summary
x %>% sd
hist(x)

plot(density(rnorm(1000)))
plot(density(rnorm(10000)))
plot(density(rnorm(1000000)))

ggplot() + 
  aes(x = x) + 
  geom_histogram()

dnorm(0) # probability at x=0
pnorm(0) # area upto x=0
qnorm(0.5) # area=0.5 till what value of x

#### student's t-distribution ####
set.seed(2022)
x = rt(n=1000, df = 10)
x %>% summary
x %>% sd

hist(x, freq=FALSE, breaks = "Scott")

plot(density(x))
plot(density(rt(n=1000000, df = 10)))


# Plot distribution - varying DoF
par(mfrow=c(3,3))
for(df in 1:9){
  x = rt(n=1000000, df = df)
  plot(density(x))
  
  # ggplot() + 
  #   aes(x = x) + 
  #   geom_histogram(aes(y = ..density..), colour = 1, fill = "white") +
  #   geom_density() %>% 
  #   print
}
par(mfrow=c(1,1))

# Plot distribution - varying n
par(mfrow=c(2,4))
for(i in 1:7){
  x = rt(n=10^i, df = 5)
  plot(density(x))
  
  # ggplot() + 
  #   aes(x = x) + 
  #   geom_histogram(aes(y = ..density..), colour = 1, fill = "white") +
  #   geom_density() %>% 
  #   print
}
par(mfrow=c(1,1))

# dt returns probability density (y axis of hist) corresponding to a point on x-axis
plot(density(x))
dt(2, df = 10)

# pt returns area under the t-curve. q=2 yields area from x=-Inf to x=2
# Cumulative probability
pt(q=2, df = 10)
pt(q=4, df = 10)  # 0.9987408
pt(q=Inf, df = 10)
pt(q=4, df = 10, lower.tail = F) # 1-pt(q=4, df = 10)

# qt returns x-value corresponfing area under the t-curve.
qt(p=0.9987408, df = 10) # returns 4
qt(p=pt(q=4, df = 10), df = 10) # returns 4, same as earlier

#### binomial distribution ####
# n: number of experiments you run
# size: number of trials in each experiment
# prob: prob of success in each trial
# rbinom output: #successful trials in each experiment. always less than size
rbinom(n=2, size=5, prob=0.1)
rbinom(n=10, size=2, prob=0.5)

rbinom(n=7, size=100, prob=0.5) # 100 coins tossed daily, for 7 days. heads=success
# returns how many heads each day

sample(c("H","T"), 10, replace = T)

set.seed(2022)
# prob of observing 2 successess in 5 trials. prob of success in each trial is 0.1
# dbinom returns f(x): probability density function
x = dbinom(x=2, size=5, prob=0.1)
x 
dbinom(x=5, size=10, prob=0.5)
dbinom(x=0, size=200, prob=0.02)

dbinom(x=0, size=200, prob=1)
dbinom(x=200, size=200, prob=1)
dbinom(x=100, size=200, prob=1)
dbinom(x=199, size=200, prob=1)
dbinom(x=c(0,50,100,199,200), size=200, prob=1)

dist = dbinom(x=0:40, size = 40, prob = 0.5)
plot(0:40, dist, type = "l")
mean(dist)
var(dist) 
40*0.5
40*0.5*(1-0.5)


pbinom(2,5,0.1) # prob of observing 2 successes or fewer


pbinom(3,5,0.1) - (dbinom(0,5,0.1) + dbinom(1,5,0.1) + dbinom(2,5,0.1) + dbinom(3,5,0.1))
dbinom(3,5,0.1) # is same as
pbinom(3,5,0.1) - pbinom(2,5,0.1)

set.seed(2022)
x = dbinom(x=2, size=5, prob=0.1)
x %>% mean # n*p
var(x) # n*p*(1-p)


# problem 1. Let X represent the number of sixes in 10 rolls of a fair die.  
# Simulate 50 runs of this probability experiment. 
set.seed(2022)
rbinom(50, 10, 1/6)
rbinom(50, 10, 1/6) %>% table

# problem 2. According to a recent survey, 72% of Americans prefer dogs to cats. 
# If 8 Americans are chosen at random, what is the probability that 6 prefer dogs?  
# That fewer than 6 do?
dbinom(x=6,size=8,prob=0.72)


# problem 3. A weighted coin has a 42% chance of coming up heads. 
# What is the expected number of heads in 5 tosses? 
# The standard deviation? 
# Construct a probability histogram for X, the number of heads in 5 tosses.
x_possible_values = 0:5
probs = dbinom(x_possible_values, size=5, 0.42)
weighted.mean(x_possible_values, probs) # numerically same as  5*0.42
library("tidyverse")
qplot(x_possible_values, weight= probs, geom="histogram", 
      bins = length(x_possible_values))

qplot(0:40, 
      weight = dbinom(0:40, size=40, 0.5), 
      geom="histogram", 
      bins = length(0:40))

x = rbinom(10000, 10, 0.5)
table(x) %>% plot()

#### chi-square distribution ####

#### f distribution ####

#### poisson distribution ####

#### exponential distribution ####

#### weibull distribution ####







#### Central Limit Theorem   ####
set.seed(2022)
population = 
  rnorm(10000, mean = 6, sd = 4)
# runif(10000, 1, 5)
mean(population)
var(population)

sample_means = NULL
for (i in 1:10000){
  my_sample = sample(population, 100)
  sample_means = c(sample_means, mean(my_sample))
  sample_vars = c(sample_means, var(my_sample))  
}
mean(sample_means) # mean of sample mean will be same as mean of population
var(sample_means) # var of sample mean = var(population) / sample size n

hist(sample_means, breaks = sqrt(length(sample_means)), freq = F)
lines(density(sample_means), col = "red")


