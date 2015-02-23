#!/bin/bash

if ! make release &>/dev/null; then
	echo "make fail"
	exit 1
fi

bin/release/stable_plot $@ > stab.dat
gnuplot -e "plot 'stab.dat' using 1:2 w lines title 'gpu', 'stab.dat' using 1:3 w l title 'cpu';"

