#!/opt/local/bin/gnuplot

load 'gnuplot/config.gp'

bin(x,width)=width*floor(x/width) + width/2.0

stats 'mixtures_rnd.dat' nooutput

bincount = 100
recordnum = STATS_records
iqwidth = (STATS_up_quartile - STATS_lo_quartile)
binwidth = iqwidth / bincount
xstart = STATS_lo_quartile - 0.1 * iqwidth
xend = STATS_up_quartile + 0.1 * iqwidth

# set term wxt size 1200,800

set grid x y mx my
set mxtics
set boxwidth binwidth
set xrange [xstart:xend]

set style fill solid 0.25 border 2

plot \
	'mixtures_rnd.dat' using (bin($1,binwidth)):(1 / (binwidth * recordnum)) smooth freq with boxes title 'Data' ls 1, \
	'mixtures_dat.dat' using 1:4 w l ls 1 lw 3 title 'Empirical PDF', \
	'mixtures_dat.dat' using 1:5 w l ls 4 lw 3 title 'Empirical PDF Finer', \
	'mixtures_dat.dat' using 1:3 w l ls 2 lw 3 title 'Predicted PDF', \
	'mixtures_dat.dat' using 1:2 w l ls 3 lw 3 title 'Real PDF', \

# pause -1
