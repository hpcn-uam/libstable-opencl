set term wxt size 1000,400

set xlabel "Alpha"
set ylabel "Beta"
splot 'fit.dat' using 1:2:3 with linespoints title "Likelihood"

pause -1
