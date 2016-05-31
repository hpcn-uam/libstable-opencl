#!/opt/local/bin/gnuplot

binwidth = 0.05
bin(x,width)=width*floor(x/width) + width/2.0

set boxwidth binwidth
set xrange [-5:5]

plot 'mixtures_rnd.dat' using (bin($1,binwidth)):(1.0) smooth freq with boxes
