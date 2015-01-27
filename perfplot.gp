#!/usr/bin/gnuplot

set term wxt
set ylabel 'Workgroup size'
set xlabel 'Array size'
set zlabel 'Bandwidth (mbps)'

set logscale yx 2

set grid
set hidden3d
set dgrid3d 40,40 gauss 0.75

list = system("echo $(ls *.dat)")
splot for [f in list] f u 1:2:4 title f with lines


pause -1
