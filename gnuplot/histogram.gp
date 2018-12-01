#!/opt/local/bin/gnuplot

load 'gnuplot/config.gp'

bin(x,width)=width*floor(x/width) + width/2.0

stats 'mixtures_rnd.dat' nooutput

bincount = 200
recordnum = STATS_records

percval(n) = system(sprintf("head -n %d mixtures_rnd.dat | tail -n 1", ceil(n * recordnum)))

iqwidth = (STATS_up_quartile - STATS_lo_quartile)
xstart = percval(0.02) - iqwidth * 0.1
xend = percval(0.98)
binwidth = (xend - xstart) / bincount


if (xstart < STATS_min) xstart = STATS_min
if (xend > STATS_max) xend = STATS_max

# set term wxt size 1200,800

if (exists("outfile")) set term pngcairo size 1920,1080
if (exists("outfile")) set output outfile

set grid x y mx my
set mxtics
set boxwidth binwidth
set xrange [xstart:xend]

set style fill solid 0.25 border 2

plot \
	'mixtures_rnd.dat' using (bin($1,binwidth)):(1 / (binwidth * recordnum)) smooth freq with boxes title 'Data' ls 1, \
	'mixtures_dat.dat' using 1:3 w l ls 2 lw 3 title 'Estimated PDF'
	#'mixture_initial.dat' using 1:3 w l ls 1 lw 2 title 'Empirical PDF', \
	#'mixture_initial.dat' using 1:4 w l ls 4 lw 1 title 'Empirical PDF Finer', \
	#'mixture_initial.dat' using 1:2 w l ls 5 dt 2 lw 2 title 'Initial PDF estimation', \
	#'mixtures_dat.dat' using 1:2 w l ls 3 lw 3 title 'Real PDF'

# if (!exists("outfile")) pause -1
