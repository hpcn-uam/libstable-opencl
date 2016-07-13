set term aqua enhanced font "Times-Roman, 18" dashed size 1280,1020

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

set multiplot layout 3, 2 title 'α-stable mixture estimation'

set title "α estimation"
plot for [i=0:10] 'mixture_debug.dat' u (column(3 + 5 * i)) w lines lw 2 title 'Comp. '.i

set title "β estimation"
plot for [i=0:10] 'mixture_debug.dat' u (column(4 + 5 * i)) w lines lw 2 title 'Comp. '.i

set title "μ estimation"
plot for [i=0:10] 'mixture_debug.dat' u (column(5 + 5 * i)) w lines lw 2 title 'Comp. '.i

set title "σ estimation"
plot for [i=0:10] 'mixture_debug.dat' u (column(6 + 5 * i)) w lines lw 2 title 'Comp. '.i

set title "Weight estimation"
plot for [i=0:10] 'mixture_debug.dat' u (column(7 + 5 * i)) w lines lw 2 title 'Comp. '.i

set title "MonteCarlo data"
set y2range [0:10]
set y2tics
set ytics nomirror
plot 'mixture_debug.dat' u 1 w lines lw 2 title 'Changes', \
	 'mixture_debug.dat' u 2 w lines title 'Number of components' lw 4 axes x1y2


unset multiplot
pause 4
reread
