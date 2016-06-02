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
plot \
	'mixture_debug.dat' u 2 w lines lw 2 title 'Comp. 1', \
	'mixture_debug.dat' u 6 w lines lw 2 title 'Comp. 2', \
	'mixture_debug.dat' u 10 w lines lw 2 title 'Comp. 3', \
	'mixture_debug.dat' u 14  w lines lw 2 title 'Comp. 4'

set title "β estimation"
plot \
	'mixture_debug.dat' u 3 w lines lw 2 title 'Comp. 1', \
	'mixture_debug.dat' u 7 w lines lw 2 title 'Comp. 2', \
	'mixture_debug.dat' u 11 w lines lw 2 title 'Comp. 3', \
	'mixture_debug.dat' u 15  w lines lw 2 title 'Comp. 4'

set title "μ estimation"
plot \
	'mixture_debug.dat' u 4 w lines lw 2 title 'Comp. 1', \
	'mixture_debug.dat' u 8 w lines lw 2 title 'Comp. 2', \
	'mixture_debug.dat' u 12 w lines lw 2 title 'Comp. 3', \
	'mixture_debug.dat' u 16  w lines lw 2 title 'Comp. 4'

set title "σ estimation"
plot \
	'mixture_debug.dat' u 5 w lines lw 2 title 'Comp. 1', \
	'mixture_debug.dat' u 9 w lines lw 2 title 'Comp. 2', \
	'mixture_debug.dat' u 13 w lines lw 2 title 'Comp. 3', \
	'mixture_debug.dat' u 17  w lines lw 2 title 'Comp. 4'

set title "MonteCarlo data"
set rmargin at screen 0.97
set y2range [0:2]
set y2tics
set ytics nomirror
plot 'mixture_debug.dat' u 1 w lines lw 2 title 'Changes', \
	2 * 0.999 ** x title 'Generator variance' axes x1y2 lw 4


unset multiplot
pause 4
reread
