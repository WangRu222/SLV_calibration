function psi=adi_p_b(S, z, sigma, p_initial, t_start, t_end, dt, mu, q, kappa,theta,xi,rho,alpha,v0 )



%%% this function has a boundary condition for pe^-z which is different
%%% from adi_p.m




% S is a column vector, z is a row vector, sigma is the leverage function
% independent on t, and it is also a column vector. mu kappa theta xi rho
% are parameters consistent in the note. alpha is a parameter of ADI,
% usually taken 0.5.

% p_initial is the value of p at the time t_start. If t_start=0, it can be obtained by
% Approximate_initial.m. Otherwise, use the value of p obtained in the last
% round.


t=t_start:dt:t_end;
nt=length(t);
NS=length(S);
Nz=length(z);
dS=(max(S)-min(S))/(NS-1);
dz=(max(z)-min(z))/(Nz-1);
% [T_S1, T_S2, T_z1, T_z2]=finite_difference_mat(S,z);
[T_S1, T_S2, T_z1, T_z2, T_zb1, T_zb2]=finite_difference_mat_b2(S,z,kappa,theta,xi);

p=p_initial(2:end-1,2:end-1); %let p equal to initial value
% I = trapz(S',trapz(z',p_initial,2));

psi=zeros(NS,Nz);
% impose the boundary condition that p=0 on the boundary, so we only need
% inner points of S, z and sigma
S_inner=S(2:end-1);
z_inner=z(2:end-1)';
sigma_inner=sigma(2:end-1);
[X, Y]=meshgrid(z_inner,S_inner);
for i=1:nt-1
    % F0p corresponds to F_0(P) in the (6)
    F0p=xi*rho*stackvector(sigma_inner,1).*T_S1*p*(T_z1);
    % F1 corresponds to F_1(P) in the (6), Since it can be written as p*F1,
    % here F1 is the coefficient matrix. That is why I left mutipy p when
    % computing A
    F1=-(kappa*theta-0.5*xi^2)*(T_zb1.*stackvector(exp(-z_inner)/v0,2))+kappa*T_z1+0.5*xi^2*(T_zb2.*stackvector(exp(-z_inner)/v0,2));
    % F2p corresponds to F_2(P) in the (6)
    F2p=-(mu-q)*(T_S1)*p+0.5*stackvector(sigma_inner.^2,1).*T_S1*p*diag(exp(z_inner)*v0)+0.5*stackvector(sigma_inner.^2,1).*T_S2*p*diag(exp(z_inner)*v0);
    A=p+dt*(F0p+p*F1+F2p);
    B=(A-alpha*dt*p*F1)/(eye(Nz-2)-alpha*dt*F1);
    %to compute c, we need to convert the matrix to vertor form ( Ns-2,Nz-2 to
    %Ns-2*Nz-2) 
    Ar1=repmat(T_S1,1,Nz-2);
    Ac1=mat2cell(Ar1,NS-2, repmat(NS-2,1,Nz-2));
    K1=blkdiag(Ac1{:});
    K1=sparse(K1);
    Ar2=repmat(stackvector(sigma_inner.^2,1).*T_S1,1,Nz-2);
    Ac2=mat2cell(Ar2,NS-2, repmat(NS-2,1,Nz-2));
    K2=blkdiag(Ac2{:});
    K2=sparse(K2);

    Ar4=repmat(stackvector(sigma_inner.^2,1).*T_S2,1,Nz-2);
    Ac4=mat2cell(Ar4,NS-2, repmat(NS-2,1,Nz-2));
    K4=blkdiag(Ac4{:});
    K4=sparse(K4);


    K3=[];
    for j=1:Nz-2
        K3=blkdiag(K3,exp(z_inner(j))*v0*speye(NS-2));
    end




    C=(speye((NS-2)*(Nz-2))-alpha*dt*(-(mu-q)*K1+0.5*K3*K2+0.5*K3*K4))\reshape(B-alpha*dt*F2p,[],1);
    p=reshape(C,NS-2,Nz-2);

     p(p<0)=0;

% int_p(i)=sum(sum(p*dz*dS))
% sum(sum(p*dz*dS))
% subplot(1,2,1)
% mesh(X,Y,p)
% xlim([-10 10]);
% ylim([-3 3]);
% zlim([0 2]);
% pause(0.000001);
% subplot(1,2,2)
%  mesh(X,Y,p_initial(2:end-1,2:end-1))
% xlim([-10 10]);
% ylim([-3 3]);
% zlim([0 2]);
% pause(0.000001);

end

psi(2:end-1,2:end-1)=p;
