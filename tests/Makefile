# Libstable test programs compilation Makefile
# 
# This file generates sample and test programs against the static
# version of Libstable
#
# Copyright (C) 2013. Javier Royuela del Val
#                    Federico Simmross Wattenberg
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or (at
#your option) any later version.
# 
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; If not, see <http://www.gnu.org/licenses/>.
#
#  Javier Royuela del Val.
#  E.T.S.I. Telecomunicación
#  Universidad de Valladolid
#  Paseo de Belén 15, 47002 Valladolid, Spain.
#  jroyval@lpi.tel.uva.es    
#
CC = gcc
#CFLAGS  =  -O3 -march=native -ffast-math -DHAVE_INLINE -I../stable/
CFLAGS = -O0 -g -pg -I../stable
LDFLAGS = # -g -pg
LIBS    = -lm -pthread -lgsl -lgslcblas

.PHONY : all clean distclean

stable_test: stable_test.o print_arch.o print_arch.h ../stable/libstable.so ../stable/libstable.a
	@echo -ne '\n$@:\n\t'
	$(CC) $(LDFLAGS) $< print_arch.o ../stable/libstable.a -o $@ $(LIBS)
	@echo

stable_array: stable_array.o
	@echo -ne '\n$@:\n\t'
	$(CC) $(LDFLAGS) $< ../stable/libstable.a -o $@ $(LIBS)
	@echo
		
fittest: fittest.o
	@echo -ne '\n$@:\n\t'
	$(CC) $(LDFLAGS) $< ../stable/libstable.a -o $@ $(LIBS)
	@echo

example: example.o
	@echo -ne '\n$@:\n\t'
	$(CC) $(LDFLAGS) $< ../stable/libstable.a -o $@ $(LIBS)
	@echo

%.o : %.c
	@echo -ne '$@:\t'
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	@echo -ne '$@:\n\t'
	rm -f *.o
	@echo	

../stable/libstable.a:
	make all -C ../stable/
	
all: clean stable_test stable_array fittest
