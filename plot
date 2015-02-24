#!/bin/bash

if ! make release &>/dev/null; then
	echo "make fail"
	exit 1
fi

bin/release/stable_plot $@ > stab.dat

