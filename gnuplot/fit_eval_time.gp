#!/usr/bin/gnuplot

set term aqua

set ylabel "β"
set xlabel "α"
set zlabel "ms per fit" rotate

set hidden3d
set dgrid3d 50,50 qnorm 2


splot data using 1:2:5 with lines title "Time per fit"

pause -1
