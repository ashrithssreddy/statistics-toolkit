library("dplyr")
rm(list=ls());cat("\014")

#### Get Normal Distribution corresponding to given x values ####
x <- seq(-5,5,by = .0002)
y <- dnorm(x, mean = 0, sd = 1)
plot(x,y)

# Incorrect to calculate this way
# mean(y)
# sd(y)
# mean_x = sum(x)/length(x)
# sd_x =  sqrt(    sum((x - mean(x))^2)/length(x)   )

# x = c(1,2,3,4,5)
# y = c(0.2, 0.4, 0.1, 0.2, 0.1)
# x - mean(x)


mean_x = sum(x*y)/sum(y)
sd_x = sqrt(sum(y*((x - mean_x)^2))/(sum(y)))

my_dnorm = function(x, mean = 0, sd = 1){
  y = (1/(2*pi*sd^2))*exp(-(x-mean)*(x-mean)/(2*sd^2))
}
my_y <- my_dnorm(x, mean = 0, sd = 1)
# plot(x,my_y)
mean_x = sum(x*my_y)/sum(my_y)
sd_x = sqrt(sum(y*((x - mean_x)^2))/(sum(y)))