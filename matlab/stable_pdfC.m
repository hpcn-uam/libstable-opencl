function pdf=stable_pdfC(parms,x,param)
% pdf = stable_pdfC(params,x,p)
% Code for computing the PDF of an alpha-estable distribution.
% Expresions presented in [1] are employed.
%    params: parameters of the alpha-estable distributions.
%                 params=[alpha,beta,sigma,mu];
%    x:      vector of evaluataion points.
%    param:  parameterization employed (view [1] for details)
%             0:   mu=mu_0
%             1:   mu=mu_1
%
% [1] Nolan, J. P. Numerical Calculation of Stable Densities and
%     Distribution Functions Stochastic Models, 1997, 13, 759-774
%
% Copyright (C) 2013. Javier Royuela del Val
%                     Federico Simmross Wattenberg

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

dist=calllib('libstable','stable_create',parms(1),parms(2),parms(3),parms(4),param);

calllib('libstable','stable_set_THREADS',0);
calllib('libstable','stable_set_relTOL',1e-12);
calllib('libstable','stable_set_absTOL',1e-12);

n=length(x);
pdf=zeros(1,n);
[~,~,pdf,~]=calllib('libstable','stable_pdf',dist,x,n,pdf,[]);

calllib('libstable','stable_free',dist);

end
