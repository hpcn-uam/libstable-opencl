#!/bin/bash

function usage() {
	echo "usage: store_mixture_results.sh testname"
	exit 1
}

if [ "$1" == "-h" ] || [ -z "$1" ]; then
	usage
fi

testname="$1"

mkdir -p results/${testname}

mv mixture_{birth,debug,initial,split}.dat mixtures_{dat.dat,rnd.dat,summary.txt} results/${testname}
