# Libstable compilation Makefile
# 
# This filee generates both shared and static versions of Libstable
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
#CFLAGS  = -fPIC -O3 -march=native -ffast-math -DHAVE_INLINE #-g -pg# -DDEBUG
CFLAGS = -O0 -fPIC -g -pg
SRCS = stable_dist.c stable_pdf.c stable_cdf.c stable_rnd.c stable_inv.c stable_integration.c methods.c mcculloch.c stable_fit.c stable_common.c stable_koutrouvelis.c
SRCSV = stable_dist_vect.c
OBJECTS = $(SRCS:.c=.o)
OBJECTSV = $(SRCSV:.c=.o)
HEADERS = stable_api.h stable_integration.h methods.h mcculloch.h
LIBS =  -lgsl -lgslcblas -lm

.PHONY : all clean distclean

libstable.so: $(OBJECTS) $(OBJECTSV) $(HEADERS)
	@echo '$@:'
	@echo -n '  '
	$(CC) -shared $(OBJECTS) $(OBJECTSV) -o $@ $(LIBS)

libstable.a: $(OBJECTS) $(OBJECTSV) $(HEADERS)
	@echo '$@:'
	@echo -n '  '
	ar -rv $@ $(OBJECTS) $(OBJECTSV)

%.o : %.c
	@echo -ne '$@:\t'
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	@echo -ne '$@:\n\t'
	rm -f *.o
	@echo	

distclean:
	@echo -ne '$@:\n\t'
	rm -f *.o
	@echo

all: clean libstable.so libstable.a distclean
