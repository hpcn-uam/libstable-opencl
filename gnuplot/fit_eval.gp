#!/usr/bin/gnuplot

set term wxt

set title "Grid fit bias"

set xlabel "$\\alpha$" offset 0,-1
set ylabel "$\\beta$" offset 0,-1
set zlabel "Bias" rotate

set view 75, 30
set ticslevel 0

set hidden3d
set dgrid3d 50,50 qnorm 1.4

set xtics border out offset 0,-0.5
set ytics border out offset 0,-1
set ztics border out

set grid xtics ytics ztics

set yrange [0:1]

splot data using 1:2:($6-$1) with lines title "$\\alpha$ estimation bias",\
	data using 1:2:($8-$2) with lines title "$\\beta$ estimation bias", \
	data using 1:2:($10-$4) with lines title "$\\sigma$ estimation bias", \
	data using 1:2:($12-$3) with lines title "$\\mu$ estimation bias"

pause -1

