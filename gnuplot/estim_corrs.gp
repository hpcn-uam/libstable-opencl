if (exists("outfile")) set term pngcairo size 1920,1080
if (exists("outfile")) set output outfile
if (!exists("outfile")) set term aqua enhanced font "Times-Roman, 18" dashed size 800,800

burnin = 500

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

set multiplot layout 4, 4 title 'α-stable estimation | Parameter value scatterplot'

set style fill transparent solid 0.1 noborder
set style circle radius screen 0.001

array pars[4]
pars[1] = "α"
pars[2] = "β"
pars[3] = "γ"
pars[4] = "δ"

do for [row=1:4] {
	do for [col=1:4] {
		if (row == col) {
			set label 1 pars[row] at 0.5,0.5 font ",40"
			unset border
			unset tics
			unset mxtics
			unset mytics
			unset xlabel
			unset ylabel
			plot [0:1] [0:1] NaN
		} else {
			unset label 1
			set border
			set tics
			set mxtics
			set mytics
			set xlabel pars[row]
			set ylabel pars[col]
			plot for [i=0:10] 'mixture_debug.dat' every ::burnin u (column(2 + row + 5 * i)):(column(2 + col + 5 * i)) w circles lw 2 title 'Comp. '.i
		}
	}
}
