function V= OptionPrice2_put(x,z,T_1,T_2,nt, L, psi, r, q, V_initial, y_0)
% diagnoal matrix
dx=(max(x)-min(x))/(length(x)-1);
dt=(T_2-T_1)/(nt-1);
dz=(max(z)-min(z))/(length(z)-1);
C=y_0*(psi*exp(z)*dz)./(psi*ones(length(z),1)*dz);
C=inpaint_nans(C,3);
C(C<0)=0;

M=zeros(length(x),length(x));
M(1,1)=1; M(1,2)=-1;


M(end,end)=1;M(end,end-1)=-1;
M=sparse(M);
A=1+L(2:end-1).^2.*C(2:end-1)*dt/(dx^2)+dt*q;
A_plus=(r-q+0.5*L(2:end-1).^2.*C(2:end-1))*dt/2/dx-0.5*L(2:end-1).^2.*C(2:end-1)*dt/(dx^2);
A_minus=-(r-q+0.5*L(2:end-1).^2.*C(2:end-1))*dt/2/dx-0.5*L(2:end-1).^2.*C(2:end-1)*dt/(dx^2);
M(2,1)=A_minus(1);
M(end-1,end)=A_plus(end);
M(2:end-1,2:end-1)=spdiags([A_plus A A_minus],-1:1,length(x)-2,length(x)-2)';

V=V_initial;

for i=1:nt-1
    f=V;
%     f(1)=exp(x(1))*exp(-r*(T_1+i*dt))*dx; 
% here we use taylor expension up to 4 as the boundary condition so that we
% can make sure no arbitrage
    f(1)=0;
    f(end)=exp(x(end))*exp(-r*(T_1+i*dt))*dx+1/2*exp(x(end))*exp(-r*(T_1+i*dt))*dx^2+1/6*exp(x(end))*exp(-r*(T_1+i*dt))*dx^3+ ...
        1/24*exp(x(end))*exp(-r*(T_1+i*dt))*dx^4+1/120*exp(x(end))*exp(-r*(T_1+i*dt))*dx^5;
    V=M\f;
    V(V<0)=0;
end
end