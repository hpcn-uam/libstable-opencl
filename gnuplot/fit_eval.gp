#!/usr/bin/gnuplot

set term wxt

set xlabel "α"
set ylabel "β"

set hidden3d
set dgrid3d 50,50 qnorm 2

set xtics border out
set ytics border out

splot data using 1:2:($6-$1) with lines title "α estimation bias",\
	data using 1:2:($8-$1) with lines title "β estimation bias"

pause -1
