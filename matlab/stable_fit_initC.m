function parms=stable_fit_initC(data)
% params=stable_fit_initC(data)
% McCulloch [1] alpha-stable parameters estimation.
%    data:   vector of random data
%    params: parameters of the alpha-estable distributions.
%                 params=[alpha,beta,sigma,mu];
%
% Copyright (C) 2013. Javier Royuela del Val
%                     Federico Simmross Wattenberg
%
% [1] John H. McCulloch. Simple consistent estimators of stable distribution pa-
%     rameters. Communications in Statistics – Simulation and Computation,
%     15(4):1109–1136, 1986.

% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 3 of the License, or (at
% your option) any later version.
% 
% This program is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
% General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program; If not, see <http://www.gnu.org/licenses/>.
%
%
%  Javier Royuela del Val.
%  E.T.S.I. Telecomunicación
%  Universidad de Valladolid
%  Paseo de Belén 15, 47002 Valladolid, Spain.
%  jroyval@lpi.tel.uva.es    
%

dist=calllib('libstable','stable_create',1,0,1,0,0);
calllib('libstable','stable_set_THREADS',0);
calllib('libstable','stable_set_relTOL',1e-8);
calllib('libstable','stable_set_absTOL',1e-8);

n=length(data);
pnu_c=0;pnu_z=0;

calllib('libstable','stable_fit_init',dist,data,n,pnu_c,pnu_z);
parms=[dist.Value.alfa dist.Value.beta dist.Value.sigma dist.Value.mu_0];

calllib('libstable','stable_free',dist);
