#!/usr/bin/gnuplot

set termoption dash;
set term aqua size 1000, 400;
plot 	'stab.dat' using 1:2 w lines title 'GPU', \
		'stab.dat' using 1:3 w lines lt 3 lc 3 title 'CPU';
pause 2
replot
reread
