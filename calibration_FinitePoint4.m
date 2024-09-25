function [Leverage_surf, impv_surf, x_L]=calibration_FinitePoint4(x, z, Maturity, L_initial, market_impv, r,q,kappa,theta,xi,rho,alpha,y0, epsilon,epsilon2, step_size, NN, Market_strikes, Market_maturities)

x_L=x;
dx=(max(x)-min(x))/(length(x)-1);
dz=(max(z)-min(z))/(length(z)-1);

dense_para_x=20;
x_dense=repelem(x', dense_para_x)';
x_increased=x(1):(x(2)-x(1))/dense_para_x:x(2);
x_increased(end)=[];
x_increased=x_increased-x(1);
x_increased=repmat(x_increased,1,length(x));
x_sparse=x_dense;
x_dense=x_dense+x_increased';
x_dense(end-dense_para_x+2:end)=[];
x_sparse(end-dense_para_x+2:end)=[];
dx_dense=dx/dense_para_x;

dense_para_z=1;
z_dense=repelem(z', dense_para_z)';
z_increased=z(1):(z(2)-z(1))/dense_para_z:z(2);
z_increased(end)=[];
z_increased=z_increased-z(1);
z_increased=repmat(z_increased,1,length(z));
z_dense=z_dense+z_increased';
dz_dense=dz/dense_para_z;

Maturity=[0; Maturity];


[initial_p,dt]= initial_by_bisection(x_dense,z_dense,1,r,q, kappa,theta,xi,rho,y0);
p=initial_p;
nT=length(Maturity);
%boundary condition for L
lb=-1; rb=1;


V_initial=max(1-exp(x_dense),0);
V_initial_put=max(exp(x_dense)-1,0);

 L=L_initial(:,1);


Leverage_surf=zeros(length(x),length(Maturity)-1);
impv_surf=zeros(length(x),length(Maturity)-1);




for i=1:nT-1
    t_start=Maturity(i);
    t_end=Maturity(i+1);

    v=zeros(length(x),NN+1);
    s=zeros(length(x),NN+1);

    beta_1=0.9;
    beta_2=0.9;

    for n=1:NN

        impv=zeros(length(x),1);
        vega=zeros(length(x),1);
        L_dense=interp1(x,L,x_dense,'spline');
        L_old=L;
        G=log(L);
        


        if t_end<dt
            [initial_p,dt]= initial_by_bisection(x_dense,z_dense,1,r,q, kappa,theta,xi,rho,y0);
            p=initial_p;
            V=OptionPrice2(x_dense,z_dense,t_start,t_end,5,L_dense, initial_p, r, q, V_initial, y0);
            V_put=OptionPrice2_put(x_dense,z_dense,t_start,t_end,5,L_dense, initial_p, r, q, V_initial_put, y0);

        elseif t_start<dt && t_end>dt
            [initial_p,dt]= initial_by_bisection(x_dense,z_dense,1,r,q, kappa,theta,xi,rho,y0);
            V=OptionPrice2(x_dense,z_dense,t_start,dt,5,L_dense, initial_p, r, q, V_initial, y0);
            V_put=OptionPrice2_put(x_dense,z_dense,t_start,dt,5,L_dense, initial_p, r, q, V_initial_put, y0);

            V_middle=V;
            V_middle_put=V_put;

            p=adi_p_b(x_dense, z_dense, L_dense, initial_p, dt, t_end, (t_end-dt)/1, r,q,kappa,theta,xi,rho,alpha,y0 );
            V=OptionPrice2(x_dense,z_dense,dt,t_end,5,L_dense, p, r, q, V_middle, y0);
            V_put=OptionPrice2_put(x_dense,z_dense,dt,t_end,5,L_dense, p, r, q, V_middle_put, y0);

        else
            p=adi_p_b(x_dense, z_dense, L_dense, initial_p, t_start, t_end, (t_end-t_start)/1, r,q,kappa,theta,xi,rho,alpha,y0 );
            V=OptionPrice2(x_dense,z_dense,t_start,t_end,5,L_dense, p, r, q, V_initial, y0);
            V_put=OptionPrice2(x_dense,z_dense,t_start,t_end,5,L_dense, p, r, q, V_initial_put, y0);


        end
%        
psi=p*exp(-r*(Maturity(i+1)));



        if i>2
            G_T=(G-log(Leverage_surf(:,i-1)))/(t_end-t_start);
        else
            G_T=zeros(length(x),1);
        end
%     visulization
    subplot(1,2,1)
    plot(x_L, G, 'r*')
    subplot(1,2,2)
    plot(x_L, L, 'b*')


    pause(0.000001);

    sign1=zeros(length(x),1);



    for j=1:length(x)


% using out of the money option to compute impv
        if x(j)>0
            if V(x_dense==x(j))>=exp(-q*t_end)
                sign1(j)=1;
                impv(j)=nan;
            else
                impv(j)=blsimpv(1,exp(x(j)),r,t_end,V(x_dense==x(j)),'Yield',q,'Class', {'Call'});
            end
        else
            if V_put(x_dense==x(j))>=exp(-r*t_end)*exp(x(j))
                sign1(j)=1;
                impv(j)=nan;
            else
                impv(j)=blsimpv(1,exp(x(j)),r,t_end,V_put(x_dense==x(j)),'Yield',q,'Class', {'Put'});
            end
        end

            
        if isnan(impv(j))
            continue
        else
            vega(j)=blsvega(1,exp(x(j)),r,t_end, impv(j),q);
        end
        
        if vega(j)==0
                vega(j)=1e-6;
        end


    end






    




             gradient=zeros(length(x),1);
             for t=1:length(Market_strikes)
                 for h=1:length(Market_maturities)

                     if market_impv(t,h)==0
                         continue
                     else
                             int_psi=y0*(psi*exp(z_dense))*dz_dense;
                             int_psi=int_psi(x_dense==x_sparse);
                             gradient=gradient+exp(x+2*G).*(impv-market_impv(t,h))./vega.*int_psi./(1e-6+(exp(x)-Market_strikes(t)).^2+(t_end-Market_maturities(h)).^2).^4;

                     end
                 end
             end

              gradient(2:end-1)=gradient(2:end-1)-epsilon*exp(-2*x(2:end-1)).*((G(3:end)+G(1:end-2)-2*G(2:end-1))/(dx^2)-(G(3:end)-G(1:end-2))/(2*dx))+epsilon2*G_T(2:end-1);
             gradient(isinf(gradient))=sign(gradient(isinf(gradient)));
             gradient(isnan(gradient))=0;
             gradient=gradient+1e-2*sign1;
             gradient=inpaint_nans(gradient,3);

             v(:,n+1)=beta_1*v(:,n)+(1-beta_1)*gradient;
             s(:,n+1)=beta_2*s(:,n)+(1-beta_2)*gradient.^2;
             v_modefied=v(:,n+1)/(1-beta_1^(n+1));
             s_modefied=s(:,n+1)/(1-beta_2^(n+1));
             g_modefied=(step_size*v_modefied)./(sqrt(s_modefied)+10^(-3));



             G=G-g_modefied;



    G(1)=-lb*dx+G(2);
    G(end)=rb*dx+G(end-1);
% limit the maximum and minimum of the leverage function
    G(G<-10)=-10;
    G(G>3)=3;

    L=exp(G);
    err_L=norm(L_old-L);
    err_G=abs(L_old-L);
% constrain the error within a small interval containing the market points
    err_G=max(err_G(exp(x)>=Market_strikes(1)-0.5 & exp(x)<=Market_strikes(end)+0.5));








    if err_G<1e-1
        fprintf('Maturity:%d, inter:%d, error_L:%d \n', t_end,n,err_L)
        Leverage_surf(:,i)=L_old;
        impv_surf(:,i)=impv;
        V_initial=V;
        V_initial_put=V_put;

        initial_p=p;
        break
    end
    if n==NN
        fprintf('Not converge')
        Leverage_surf(:,i)=L_old;
        impv_surf(:,i)=impv;
        V_initial=V;
        initial_p=p;
        V_initial_put=V_put;
    end

    end
end



    





