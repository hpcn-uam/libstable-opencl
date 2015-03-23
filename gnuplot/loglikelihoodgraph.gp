set term x11 size 1000,400

set xlabel "Alpha"
set ylabel "Beta"
set zlabel "- log(likelihood)"
splot 'fit.dat' using 1:2:3 with linespoints

pause -1
