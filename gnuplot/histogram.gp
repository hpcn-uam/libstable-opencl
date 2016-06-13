#!/opt/local/bin/gnuplot

binwidth = 0.05
bin(x,width)=width*floor(x/width) + width/2.0

set boxwidth binwidth
set xrange [-5:5]

plot 'mixtures_rnd.dat' using (bin($1,binwidth)):(1 / 250.0) smooth freq with boxes title 'Data', \
	'mixtures_dat.dat' using 1:2 w l lw 3 title 'Real PDF', \
	'mixtures_dat.dat' using 1:3 w l lw 3 title 'Predicted PDF', \
	'mixtures_dat.dat' using 1:4 w l lw 3 title 'Empirical PDF'
