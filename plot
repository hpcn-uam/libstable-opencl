#!/bin/bash

if ! make release &>/dev/null; then
	echo "make fail"
	exit 1
fi

bin/release/stable_plot $@ > stab.dat.1
cat stab.dat.1 | grep "V" | tr -d 'V' | sort -n > V.dat.b
cat stab.dat.1 | grep "G" | tr -d 'G' | sort -n > G.dat.b
cat stab.dat.1 | grep -v "V" | grep -v "G" | sort -n > stab.dat.b
mv V.dat.b V.dat
mv G.dat.b G.dat
mv stab.dat.b stab.dat

