function [status parms]=stablefitwholeC(data,varargin)
% cdf=stablecdfC(parms,x)
%
% Estima los parametros de una distribucion estable a partir de los datos dados
% Metodo de Javier Royuela del Val [Royuela, J., PFC, UVa, 2011] (basado en
% MLE + McCulloch)
%
% parms=[alfa_hat beta_hat gamma_hat delta_hat]
% data=vector de datos

%los parametros especificados al crear la distribucion no se tienen en
%cuenta al estimar pero deben ser validos
%dist=calllib('libstable','stable_create',1,0,1,0,0);


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
calllib('libstable','stable_set_relTOL',1e-6);
calllib('libstable','stable_set_absTOL',1e-8);

status=calllib('libstable','stable_fit_whole',dist,data,n);

parms=[dist.Value.alfa dist.Value.beta dist.Value.sigma dist.Value.mu_0];

calllib('libstable','stable_free',dist);




