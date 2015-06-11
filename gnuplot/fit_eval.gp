#!/usr/bin/gnuplot

set term cairolatex size 15cm,10cm
set output "fit_eval_bias.tex"

set xlabel "α"
set ylabel "β"

set hidden3d
set dgrid3d 50,50 qnorm 1.4

set xtics border out
set ytics border out

splot data using 1:2:(abs($6-$1)) with lines title "alpha estimation bias",\
	data using 1:2:(abs($8-$2)) with lines title "beta estimation bias"
