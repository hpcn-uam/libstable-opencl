#!/opt/local/bin/gnuplot

load 'gnuplot/config.gp'

stats 'mixtures_rnd.dat' nooutput

recordnum = STATS_records

# set term wxt size 1200,800

if (exists("outfile")) set term pngcairo size 1920,1080
if (exists("outfile")) set output outfile

set grid x y mx my
set mxtics

set style fill solid 0.25 border 2

plot \
	'mixtures_rnd.dat' using 1:(1./recordnum) smooth cumulative title 'ECDF' ls 1 lw 2, \
	'mixtures_dat.dat' using 1:4 w l ls 2 lw 2 title 'Predicted CDF', \
	#'mixtures_dat.dat' using 1:4 w l ls 1 lw 3 title 'Empirical PDF', \
	#'mixtures_dat.dat' using 1:5 w l ls 4 lw 3 title 'Empirical PDF Finer', \
	#'mixtures_dat.dat' using 1:2 w l ls 3 lw 3 title 'Real PDF'

if (!exists("outfile")) pause -1
