#!/usr/bin/gnuplot

set termoption dashed
set term pngcairo size 30cm,13cm dashed

set output "pdf_performance.png"

set title "Multiple points GPU PDF performance test"
set xrange[0:800]

set xlabel "N. points"
set ylabel "Milliseconds"
set y2label "Milliseconds per point"

set ytics nomirror
set y2tics

set grid x y2

set key center top

plot	data u 1:2 lt 19 lc 4 w l title "GPU - Total time"  axes x1y1, \
	 	data u 1:3 lt 1 lc 1 w l title "GPU - Time per point" axes x1y2, \
	 	data u 1:4 lt 19 lc 2 w l title "CPU - Total time"  axes x1y1, \
	 	data u 1:5 lt 1 lc 3 w l title "CPU - Time per point" axes x1y2

