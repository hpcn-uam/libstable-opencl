#!/usr/bin/env Rscript

results = read.table("GRD_GPU.dat")
time = results$V5

alpha_bias = results$V6 - results$V1
beta_bias = results$V8 - results$V2
sigma_bias = results$V10 - results$V4
mu_bias = results$V12 - results$V3

X11()

print("Time distribution quantiles: ")
print(quantile(time, c(.10, .25, .50, .75, .90, .95)))

boxplot(alpha_bias, beta_bias, mu_bias, sigma_bias, horizontal = TRUE, names = c("alpha", "beta", "mu", "sigma"), main = "ML Estimation Bias", boxwex = 0.2, height = 50, cex=0.5)

message("Press Return To Continue")
invisible(readLines("stdin", n=1))
