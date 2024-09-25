function [T_S1, T_S2, T_z1, T_z2, T_zb1, T_zb2]=finite_difference_mat_b2(S,z,kappa,theta,xi)

%the construction are consistent with section 3 Space discretisation in the
%note.

Ns=length(S);
Nz=length(z);

dS=S(2:end)-S(1:end-1);
dS_i_minus=dS(1:end-1);
dS_i=dS(2:end);

c_S_minus=-dS_i./(dS_i_minus.*(dS_i_minus+dS_i));
c_S=(dS_i-dS_i_minus)./(dS_i_minus.*dS_i);
c_S_plus=dS_i_minus./(dS_i.*(dS_i_minus+dS_i));
c_S(1)=c_S(1)+c_S_minus(1);
c_S(end)=c_S(end)+c_S_plus(end);

s_S_minus=2./(dS_i_minus.*(dS_i_minus+dS_i));
s_S=-2./(dS_i_minus.*dS_i);
s_S_plus=2./(dS_i.*(dS_i_minus+dS_i));
s_S(1)=s_S(1)+s_S_minus(1);
s_S(end)=s_S(end)+s_S_plus(end);

T_S2=spdiags([s_S_plus s_S s_S_minus],-1:1,Ns-2,Ns-2)';
T_S1=spdiags([c_S_plus c_S c_S_minus],-1:1,Ns-2,Ns-2)';

dz=z(2:end)-z(1:end-1);
dz_i_minus=dz(1:end-1);
dz_i=dz(2:end);
c_z_minus=-dz_i./(dz_i_minus.*(dz_i_minus+dz_i));
c_z=(dz_i-dz_i_minus)./(dz_i_minus.*dz_i);
c_z_plus=dz_i_minus./(dz_i.*(dz_i_minus+dz_i));
%% here we give a boundary condition for pe^-z
c_z_b=c_z;
c_z_b(1)=c_z_b(1)+(1+dz_i_minus(1)-dz_i_minus(1)*2*kappa*theta/(xi^2))*c_z_minus(1);
%%%%%%%%%%%%%%%%%%%%%
c_z(1)=c_z(1)+c_z_minus(1);% 
c_z(end)=c_z(end)+c_z_plus(end);


s_z_minus=2./(dz_i_minus.*(dz_i_minus+dz_i));
s_z=-2./(dz_i_minus.*dz_i);
s_z_plus=2./(dz_i.*(dz_i_minus+dz_i));
%% here we give a boundary condition for pe^-z from zero flux condition
s_z_b=s_z;
s_z_b(1)=s_z_b(1)+(1+dz_i_minus(1)-dz_i_minus(1)*2*kappa*theta/(xi^2))*s_z_minus(1);
%%%%%%%%%%%%%%%%%%%%%%%
s_z(1)=s_z(1)+s_z_minus(1);%
s_z(end)=s_z(end)+s_z_plus(end);


T_z2=spdiags([s_z_plus s_z s_z_minus],-1:1,Nz-2,Nz-2);
T_z1=spdiags([c_z_plus c_z c_z_minus],-1:1,Nz-2,Nz-2);

T_zb2=spdiags([s_z_plus s_z_b s_z_minus],-1:1,Nz-2,Nz-2);
T_zb1=spdiags([c_z_plus c_z_b c_z_minus],-1:1,Nz-2,Nz-2);