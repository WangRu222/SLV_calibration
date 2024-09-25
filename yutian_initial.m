function initial_p= yutian_initial(S,z,sigma_initial,mu, q, kappa,theta,xi,rho,dt,v0)
% S is a column vector, z is a row vector
% sigma_initial=sigma(S0,0)
mu_S=((mu-q)-0.5*sigma_initial^2*v0)*dt;
sigma_S=sigma_initial*sqrt(v0)*sqrt(dt);
mu_z=((kappa*theta-0.5*xi^2)/v0-kappa)*dt;
sigma_z=xi*sqrt(dt/v0);
initial_p=zeros(length(S),length(z));
for i=1:length(S)
    initial_p(i,:)=1/(2*pi*sigma_S*sigma_z*sqrt(1-rho^2))*exp(-((S(i)-mu_S)^2/(sigma_S^2)+((z-mu_z).^2)./(sigma_z^2)-(2*rho*(S(i)-mu_S)*(z-mu_z)./(sigma_S*sigma_z)))/(2*(1-rho^2)));
end