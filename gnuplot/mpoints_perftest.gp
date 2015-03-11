#!/usr/bin/gnuplot

set termoption dash
set term aqua size 1000,600 0
set title "Multiple points GPU PDF performance test"

set xlabel "N. points"
set ylabel "Milliseconds"
set y2label "Milliseconds"

set xrange [0:8000]

set ytics nomirror
set y2tics

set grid x y2

set key center top

plot	'mpoints.dat' u 1:2 lt 1 lc 4 w l title "GPU - Total time"  axes x1y1, \
	 	'mpoints.dat' u 1:3 lt 1 lc 1 w l title "GPU - Time per point" axes x1y2, \
	 	'mpoints.dat' u 1:4 lt 1 lc 2 w l title "CPU - Total time"  axes x1y1, \
	 	'mpoints.dat' u 1:5 lt 1 lc 3 w l title "CPU - Time per point" axes x1y2

set term aqua 1
set xrange[0:800]
set title "Multiple points GPU PDF performance test - a closer look"
replot
