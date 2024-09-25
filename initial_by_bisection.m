function [initial_p,dt]= initial_by_bisection(S,z,sigma_initial,mu, q, kappa,theta,xi,rho,v0)

dt1=1e-5;
dt2=0.1;
er=1e-10;
int_diff=1;
dS=(max(S)-min(S))/(length(S)-1);
dz=(max(z)-min(z))/(length(z)-1);
dt=dt1;
t=1;
while abs(int_diff)>er
    p=yutian_initial(S,z,sigma_initial,mu, q, kappa,theta,xi,rho,dt,v0);
    int_p=0;
    for i=1:length(S)-1
        for j=1:length(z)-1
            int_p=int_p+1/4*(p(i,j)+p(i+1,j)+p(i,j+1)+p(i+1,j+1))*dz*dS;
        end
    end
    int_diff=int_p-1;
    if int_diff<0
        dt2=dt;
    else
        dt1=dt;
    end
dt=(dt1+dt2)/2;
t=t+1;
if t>15
    fprintf('The partition is too sparse');
    return
end
end
p(p<0)=0;
initial_p=p;


