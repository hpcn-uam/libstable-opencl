function [parms status]=stable_fit_koutrouvelisC(data,varargin)
% [params status]==stable_fit_koutrouvelisC(data,p0)
% Koutrouvelis [1] estimation of alpha-stable parameters.
%    data:  vector of random data
%    p0:    initial guess of parameters estimation. If none is provided,
%           McCulloch estimation is done before ML.
%    params = [alpha,beta,sigma,mu_0]. Estimated parameters.
%    status : status>0 indicates some error on the iterative procedure.
%
% Copyright (C) 2013. Javier Royuela del Val
%                     Federico Simmross Wattenberg
%
% [1] Ioannis A. Koutrouvelis. An iterative procedure for the estimation of the
%     parameters of stable laws. Communications in Statistics - Simulation and
%     Computation, 10(1):17–28, 1981.

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

pnu_c=0;pnu_z=0;
n=length(data);

if ~isempty(varargin)
    p0=varargin{1};
    dist=calllib('libstable','stable_create',p0(1),p0(2),p0(3),p0(4),0); 
else
    dist=calllib('libstable','stable_create',1,0,1,0,0);
    calllib('libstable','stable_fit_init',dist,data,n,pnu_c,pnu_z);
end

calllib('libstable','stable_set_THREADS',0);
calllib('libstable','stable_set_relTOL',1e-8);
calllib('libstable','stable_set_absTOL',1e-8);

status=calllib('libstable','stable_fit_koutrouvelis',dist,data,n);

parms=[dist.Value.alfa dist.Value.beta dist.Value.sigma dist.Value.mu_0];

calllib('libstable','stable_free',dist);
