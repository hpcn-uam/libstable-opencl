bin(x,width)=width*floor(x/width) + width/2.0

stats 'mixtures_rnd.dat' nooutput

bincount = 100
burnin = 500
thinning = 20
recordnum = STATS_records

percval(n) = system(sprintf("head -n %d mixtures_rnd.dat | tail -n 1", ceil(n * recordnum)))

iqwidth = (STATS_up_quartile - STATS_lo_quartile)
xstart = percval(0.02) - iqwidth * 0.1
xend = percval(0.95)
binwidth = 0.01
binwidthgd = 0.005

if (xstart < STATS_min) xstart = STATS_min
if (xend > STATS_max) xend = STATS_max

if (exists("outfile")) set term pngcairo size 1400,1000
if (exists("outfile")) set output outfile
if (!exists("outfile")) set term aqua enhanced font "Times-Roman, 18" dashed size 800,1000

set style fill solid 0.25 border 2

### Tics

set xtics
set ytics
set mxtics
set mytics

### Grids

set style line 102 lc rgb '#808080' lt 3 lw 1
set style line 103 lc rgb '#A0A0A0' lt 3 lw 0.5

set grid xtics mxtics ls 102, ls 103
set grid ytics mytics ls 102, ls 103

unset y2tics

set key off

set xrange [*:*]

set multiplot layout 4, 3 title 'α-stable MCMC diagnostics'

set title "α trace"
set xlabel "Iteration"
unset logscale x
plot for [i=0:10] 'mixture_debug.dat' u (column(3 + 5 * i)) w lines lw 2 title 'Comp. '.i

set title "α autocorrelation"
set logscale x
set xlabel ""
plot for [i=0:10] 'mixture_autocorr.dat' u (column(1 + 4 * i)) w lines lw 2 title 'Comp. '.i

set title "α distribution"
unset logscale x
plot for [i=0:10] 'mixture_debug.dat' every thinning::burnin u (bin(column(3 + 5 * i), binwidth)):1 smooth freq with boxes

set title "β trace"
set xlabel "Iteration"
unset logscale x
plot for [i=0:10] 'mixture_debug.dat' u (column(4 + 5 * i)) w lines lw 2 title 'Comp. '.i

set title "β autocorrelation"
set logscale x
set xlabel ""
plot for [i=0:10] 'mixture_autocorr.dat' u (column(2 + 4 * i)) w lines lw 2 title 'Comp. '.i

set title "β distribution"
unset logscale x
plot for [i=0:10] 'mixture_debug.dat' every thinning::burnin u (bin(column(4 + 5 * i), binwidth)):1 smooth freq with boxes

set title "γ trace"
set xlabel "Iteration"
unset logscale x
plot for [i=0:10] 'mixture_debug.dat' u (column(5 + 5 * i)) w lines lw 2 title 'Comp. '.i

set title "γ autocorrelation"
set logscale x
set xlabel ""
plot for [i=0:10] 'mixture_autocorr.dat' u (column(3 + 4 * i)) w lines lw 2 title 'Comp. '.i

set title "γ distribution"
unset logscale x
plot for [i=0:10] 'mixture_debug.dat' every thinning::burnin u (bin(column(5 + 5 * i), binwidthgd)):1 smooth freq with boxes

set title "δ trace"
set xlabel "Iteration"
unset logscale x
plot for [i=0:10] 'mixture_debug.dat' u (column(6 + 5 * i)) w lines lw 2 title 'Comp. '.i

set title "δ autocorrelation"
set logscale x
set xlabel ""
plot for [i=0:10] 'mixture_autocorr.dat' u (column(4 + 4 * i)) w lines lw 2 title 'Comp. '.i

set title "δ distribution"
unset logscale x
plot for [i=0:10] 'mixture_debug.dat' every thinning::burnin u (bin(column(6 + 5 * i), binwidthgd)):1 smooth freq with boxes
