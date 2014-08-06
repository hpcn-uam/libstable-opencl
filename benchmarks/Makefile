# Libstable benchmarks compilation Makefile
# 
# This file generates benchamrks programs developed to test
# Libstable precision and performance
#
# Copyright (C) 2013. Javier Royuela del Val
#                     Federico Simmross Wattenberg
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
#CFLAGS = -c -Wall -O2 -mtune=k8 -ffast-math -mfpmath=sse,387 -pg -DDEBUG
CFLAGS  = -O3 -march=native -ffast-math -DHAVE_INLINE -I../stable/ # -DDEBUG
LDFLAGS = -O3 -march=native -ffast-math -L../stable/ # -DDEBUG
LIBS    = -lm -pthread -lgsl -lgslcblas # -lstable

.PHONY : all clean distclean


stable_precision: stable_precision.o
	@echo -ne '\n$@:\n\t'
	$(CC) $(LDFLAGS) $< ../stable/libstable.a $(LIBS) -o $@
	@echo

stable_performance: stable_performance.o
	@echo -ne '\n$@:\n\t'
	$(CC) $(LDFLAGS) $< ../stable/libstable.a $(LIBS) -o $@
	@echo


%.o : %.c
	@echo -ne '$@:\t'
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	@echo -ne '$@:\n\t'
	rm -f *.o
	@echo	

all: clean stable_precision stable_performance
