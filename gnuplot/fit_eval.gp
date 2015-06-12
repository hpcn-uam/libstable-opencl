#!/usr/bin/gnuplot

set term pngcairo size 800,600
set output "fit_eval_bias.png"

set xlabel "α"
set ylabel "β"

set view 77, 30
set ticslevel 0

set hidden3d
set dgrid3d 50,50 qnorm 1.4

set xtics border out
set ytics border out

splot data using 1:2:(abs($6-$1)) with lines title "alpha estimation bias",\
	data using 1:2:(abs($8-$2)) with lines title "beta estimation bias"
