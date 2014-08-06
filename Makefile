# Libstable main Makefile
# 
# This file compiles the whole Libstable packages:
#    - Shared and static versions of the library
#    - Sample and test programs
#    - Benchmark programs
#
# Copyright (C) 2013. Javier Royuela del Val
#                    Federico Simmross Wattenberg
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or (at
# your option) any later version.
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
.PHONY : all clean stable tests benchmarks

all: stable tests benchmarks

stable:
	make -C ./stable all
	make -C ./stable clean

tests:
	make -C ./tests all
	make -C ./tests clean

benchmarks:
	make -C ./benchmarks all
	make -C ./benchmarks clean

clean:
	make -C ./stable clean
	make -C ./tests clean
	make -C ./benchmarks clean

