#!/usr/bin/gnuplot

set term pngcairo size 1920,1080
set output 'perfplot.png'
set ylabel 'Workgroup size'
set xlabel 'Array size'
set zlabel 'Mbps BW' offset 10,10

set logscale yx 2

set grid
set hidden3d
set dgrid3d 50,50 gauss 0.75

list = system("echo $(ls *.dat)")
splot for [f in list] f u 1:2:4 title f with lines

set term wxt size 1000,600
replot

pause -1
