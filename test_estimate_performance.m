y0=0.025; theta=0.059; kappa=2; rho=-0.063; xi=0.484;
r=0.025; q=0;  dt=0.001; alpha=0.5;
% y0=0.036; theta=0.021; kappa=2; rho=0.018; xi=0.288;
% r=0.0203; q=0;  dt=0.001; alpha=0.5;

T=5;
% Market_strikes=[0.5, 0.75,0.8,0.9,1,1,1,1.25,1.5];
% Market_maturities=[3/12, 6/12, 9/12, 24/12, 36/12, 60/12];
Market_strikes=[0.7925
0.8087
0.8491
0.8895
0.93
0.9704
1.0108
1.0513
1.0917
1.1321
1.1726








];
Market_maturities=[0.0438	0.1342	0.274	0.5425














];
Market_impv=[0.3742	0	0.2373	0.215
0.3536	0.2668	0.2259	0.2128
0.2942	0.2312	0.2078	0.2027
0.2415	0.2072	0.1927	0.1938
0.1912	0.1881	0.1843	0.1889
0.1606	0.1765	0.1766	0.1835
0.1553	0.1696	0.1762	0.1815
0.1652	0.1733	0.1769	0.1827
0.1875	0.1841	0.1823	0.1823
0.2283	0.2	0.188	0.1858
0.28	0.2184	0.1979	0.1878
];





tic
x=-2:0.2:2;
x=x';
z=-20:0.1:3;
z=z';
Maturity=0.01:0.02:1;
Maturity=Maturity';

%initial value of the leverage function
L_initial=1*ones(length(x),length(Maturity));
epsilon=1e-2; step_size=2e-1; NN=100; epsilon2=1e-2;

[Leverage_surf,  impv_surf,x_L]=calibration_FinitePoint4(x, z, Maturity, L_initial, ...
    Market_impv, r,q,kappa,theta,xi,rho,alpha,y0, epsilon,epsilon2,step_size, NN, Market_strikes, Market_maturities);
toc



[X,Z]=meshgrid(Maturity,exp(x_L));
[S,M]=meshgrid(Market_maturities,Market_strikes);
impv_model=interp2(X,Z,impv_surf,S,M);

error_1=abs(impv_model-Market_impv);
error_1(Market_impv==0)=0;
error_1(isnan(error_1))=0;

norm(error_1)

% visulize the performance
T_plot=[];
for i=1:length(Market_maturities)
    T_plot=[T_plot;Market_maturities(i)*ones(length(Market_strikes),1)];
end
Market_impv(Market_impv==0)=nan;

mesh(X,Z,impv_surf)
ylabel('Strike price')
xlabel('Marturity')
zlabel('Implied volatility')
hold on
scatter3(T_plot,repmat(Market_strikes,length(Market_maturities),1),Market_impv(:),'filled','k')
legend({'impv produced by the model','Market points'},'Location','northwest')
xlim([Market_maturities(1),1])
ylim([Market_strikes(1)-0.3,Market_strikes(end)+0.3])
zlim([0 0.6])
