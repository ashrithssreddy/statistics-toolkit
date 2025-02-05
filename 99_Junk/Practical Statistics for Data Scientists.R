rm(list=ls()); cat("\014")
library("ggplot2")

#### Chapter 1 - EDA ####
x = rnorm(100)
hist(x)

weights = runif(100)
weights = weights / sum(weights)
weights %>% sum

hist(weights)

mean(x)
mean(x, trim = 0.1) # trim off 10% of values each side, total 20%

matrixStats::weightedMean(x)
matrixStats::weightedMean(x, w=weights)


IQR(x)
mad(x, constant = 1) # median absolute deviation from median
(x - median(x)) %>% abs %>% median

# density plot
hist(x)
hist(x, freq=FALSE)
lines(density(x), lwd=3, col='blue')
density(x) %>% class
data.frame(density(x)$x, density(x)$y) %>% View

# hexagonal binning
ggplot(mtcars, aes(x=hp, y=mpg)) +
  stat_binhex(color='white') +
  theme_bw() +
  scale_fill_gradient(low='white', high='black') +
  labs(x='x', y='y')

#### Chapter 2 - Data and Sampling Distributions ####

## CLT
set.seed(2022)
# population = rnorm(10000)
population = runif(10000)

count_of_samples = 10000 # more samples drawn, more normal the sample_statistic
sample_size = 50 # more datapoints drawn per sample, more normal the sample_statistic

sample_statistic_mean = NULL
sample_statistic_mean_trimmed = NULL
sample_statistic_median = NULL
for(i in 1:count_of_samples){
  my_sample = sample(population, sample_size)
  sample_statistic_mean = c(sample_statistic_mean, mean(my_sample))
  sample_statistic_mean_trimmed = c(sample_statistic_mean_trimmed, mean(my_sample, trim = 0.1))
  sample_statistic_median = c(sample_statistic_median, median(my_sample))
}
hist(sample_statistic_mean, main = "sample distribution")
# hist(sample_statistic_mean_trimmed, main = "sample distribution")
# hist(sample_statistic_median, main = "sample distribution")

# s/sqrt(n):
# s = standard deviation of sample.
# ??
standard_error = sd(sample_statistic_mean)/sqrt(sample_size)
standard_error
sd(population)

## Bootstrapping
# treat the sample data you have as population
# sample (with replacement) from this population multiple times to create multiple samples
# How big is the resample? size of population
# resample how many times? 1000s of times
# sample statistic: mean, median, sd
# sd of mean value of original dataset = sd of means of bootstrapped samples
# standard error and CI of the mean
set.seed(2022)
x = rnorm(100000)
iterations = 1000
sample_size = length(x) # / 10

x_sample_means = NULL
for(iter in 1:iterations){
  x_sample = sample(x, sample_size, replace = T)
  x_sample_means = c(mean(x_sample), x_sample_means)
}
hist(x_sample_means)

# 95% CI is the interval covering 95% of bootstrapped means. 
# exclude top and bottom 2.5%
x_sample_means %>% summary
x_sample_means %>% sort %>% tail(-length(x_sample_means)*0.025) %>% min
x_sample_means %>% sort %>% head(-length(x_sample_means)*0.025) %>% max

# bias calc
x_sample_means %>% summary
x %>% summary
mean(x_sample_means) - mean(x)

# calc p-value


# bias = original mean/statistic - bootstrapped mean/statistic



## qqplot
set.seed(2022)
x = rnorm(100)
qqnorm(x)
abline(a=0, b=1)
df = data.frame(sample = qqnorm(x)$x,  theoritical = qqnorm(x)$y) %>% 
  arrange(theoritical)
View(df)

# zscore does not standardize or normalize data, just puts data on same scale as normal distribution
set.seed(2022)
x = runif(10000)
x = rpois(10000, lambda = 1)

par(mfrow = c(2,2))
hist(x, bins = 30)
hist(x- mean(x))
hist((x- mean(x))/sd(x))

hist(scale(x))
qqnorm(x)
qqnorm(scale(x))


## scaling - zscore vs min_max vs decimal
set.seed(10)
x = runif(100)
x = c(-401, -10, 201, 301, 501, 601, 701, 1000)
x %>% summary

# j = -11:15
# j = j[min(which(10^(j) > max(x)))]
j = x %>% max %>% ceiling %>% as.integer %>% nchar()
x_decimal_scaled = x/(10^j)


set.seed(10)
x = runif(100)
x %>% summary
max(x) - min(x)
sd(x)

# minmax does not handle outliers, zscore somewhat does
# sd < range => zscore squeezes the data more tightly
# range of zscore: could be anything, typically -4 to 4
# range of min-max: always 0-1
# PCA: zscore>min-max, since we are interested in the components that maximize the variance 

x_zscore = (x - mean(x)) /(sd(x)          )
x_minmax = (x - min(x) ) /(max(x) - min(x))
x_zscore %>% summary
x_minmax %>% summary

par(mfrow = c(2,2))
hist(x)
hist(x_zscore)
hist(x_minmax)

y = rnorm(100)
df = data.frame(x = x, x_zscore = x_zscore, x_minmax = x_minmax)
View(df)
cor(x_zscore, x_minmax)
cor(x,y)
cor(x_zscore, y)
cor(x_minmax, y)

(x-y)^2 %>% sum %>% sqrt
(x_zscore-y)^2 %>% sum %>% sqrt
(x_minmax-y)^2 %>% sum %>% sqrt