#include <cmath>
#include <iostream>
#include <algorithm>
#include <memory>
#include <vector>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/interpolators/cardinal_cubic_b_spline.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp> //subrange,subslice
#include <boost/numeric/ublas/matrix_proxy.hpp> //w = row(A, i); w = column(A, j);    // a row or column of matrix as a vector
#include <boost/numeric/ublas/banded.hpp>//ublas::banded_matrix<double> M(nx, nx, 1, 1);//size1,size2,lower,upper; 是否有求解三对角矩阵的线性方程组程序？
#include <boost/numeric/ublas/lu.hpp>//permutation_matrix,用于LU分解
#include <boost/numeric/ublas/triangular.hpp>//solve函数，用于求解上三角或下三角线性方程组：V =  solve(M, f, ublas::lower_tag());
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/storage.hpp>
#include <boost/numeric/ublas/assignment.hpp>//赋值符号<<==给matrix和vector赋值
#include <boost/numeric/ublas/io.hpp> //使用std::cout输出matrix和vector
#include "CumNorm.h"
//#include    <math.h>
namespace ublas = boost::numeric::ublas;
namespace interp = boost::math::interpolators;
using namespace std;
#include<Core> 
#include<SVD> 
#include<Dense> 
struct hestonParas
{
	double theta;
	double kappa;
	double VoV;   //xi
	double rho;
	double Y0;  // v0
};

template<typename _Matrix_Type_> 
_Matrix_Type_ pseudoInverse(const _Matrix_Type_ &a, double epsilon = 
    std::numeric_limits<double>::epsilon()) 
{  
    Eigen::JacobiSVD< _Matrix_Type_ > svd(a ,Eigen::ComputeThinU | Eigen::ComputeThinV);  
    double tolerance = epsilon * std::max(a.cols(), a.rows()) *svd.singularValues().array().abs()(0);  
    return svd.matrixV() *  (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint(); 
}    

void fnexp(
	ublas::vector<double> b,
    ublas::vector<double>& d
)
{

	for (size_t i = 0; i < b.size(); ++i)d(i) = exp(b(i));
}
void stackvector(
	ublas::vector<double> Y,
	double para,
	ublas::matrix<double>& matrix0
)
{
	//throw"The para must be set to 1 or 2.";
	size_t n = Y.size();
	matrix0.resize(n, n);
	if (para == 1)
	{
		for (size_t i = 0; i < n; ++i)
			for (size_t j = 0; j < n; ++j)matrix0(i, j) = Y[i];

	}
	else if (para == 2)
	{
		for (size_t i = 0; i < n; ++i)
			for (size_t j = 0; j < n; ++j)matrix0(i, j) = Y[j];
	}
	matrix0 = ublas::trans(matrix0);
}

//create the initial prob density 
void yutian_initial(
	ublas::vector<double> S,	//stock direction grid		
	ublas::vector<double> z,	//v     direction grid
	hestonParas hp,             //heston paramaters
	double r,					//interest rate
	double q,					//dividend 
	double sigma_initial,       //sigma at t = 0
	double dt,                  //
	ublas::matrix<double>& initial_p //return: initial matrix of prob density
)
{
	//S is a column vector, z is a row vector
	//sigma_initial = sigma(S0, 0)

	size_t rows = S.size();
	size_t cols = z.size();

	double mu_S = ((r - q) - 0.5 * pow(sigma_initial, 2.0) * hp.Y0) * dt;
	double sigma_S = sigma_initial * sqrt(hp.Y0) * sqrt(dt);
	double mu_z = ((hp.kappa * hp.theta - 0.5 * pow(hp.VoV, 2)) / hp.Y0 - hp.kappa) * dt;
	double sigma_z = hp.VoV * sqrt(dt / hp.Y0);
	initial_p.resize(rows, cols);

	double PI = boost::math::constants::pi<double>();
	for (size_t i = 0; i < rows; ++i)
	{
		for (size_t j = 0; j < z.size(); ++j)
		{
			initial_p(i, j) = 1 / (2 * PI * sigma_S * sigma_z * sqrt(1.0 - pow(hp.rho, 2))) *
				exp(
					-(
						pow((S(i) - mu_S), 2) / pow(sigma_S, 2) +
						(pow(z(j) - mu_z, 2) / pow(sigma_z, 2) - (2 * hp.rho * (S(i) - mu_S) * (z(j) - mu_z) / (sigma_S * sigma_z))) / (2 * (1 - pow(hp.rho, 2)))
						)
				);
		}
	}
}

void bisection_initial(
	ublas::vector<double> S,	//stock direction grid		
	ublas::vector<double> z,	//v     direction grid
	hestonParas hp,             //heston paramaters
	double r,					//interest rate
	double q,					//dividend
	double sigma_initial,       //sigma at t = 0

	double& dt,                  //return: initial value: 1e-5
	ublas::matrix<double>& initial_p //return: initial matrix of prob density
)
{
	double dt1 = 1e-5;
	double dt2 = 0.1;
	double er = 1e-10;
	double int_diff = 1.0;
	size_t nS = S.size();
	size_t nz = z.size();
	double S_max = S[nS - 1], S_min = S[0]; // 最大最小值，s,z是顺序生成的？
	double z_max = z[nz - 1], z_min = z[0];
	double dS = (S_max - S_min) / (nS - 1);
	double dz = (z_max - z_min) / (nz - 1);
	dt = dt1;
	double t = 1.0;
    //这里如果条件不加<15就会一直循环下去
	while ((int_diff) > er && t<15)
	{
		yutian_initial(S, z, hp, r, q, sigma_initial, dt, initial_p);
		double int_p = 0;

		for (size_t i = 0; i < nS - 1; ++i)
		{
			for (size_t j = 0; j < nz - 1; ++j)
			{
				int_p = int_p + (1.0 / 4.0) * (initial_p(i, j) + initial_p(i + 1, j) + initial_p(i, j + 1), initial_p(i + 1, j + 1)) * dz * dS;
			}
		}
		int_diff = int_p - 1;
		if (int_diff < 0)dt2 = dt;
		else        	dt1 = dt;

		dt = (dt1 + dt2) / 2.0;
		t = t + 1;
		if (t > 15)throw"The partition is too sparse";
	}

	for (size_t i = 0; i < initial_p.size1(); ++i)
		for (size_t j = 0; j < initial_p.size2(); ++j)
			if (initial_p(i, j) < 0)initial_p(i, j) = 0;
}
void identify_neighbors(
	ublas::matrix<double> A,
	ublas::matrix<double> nan_list,
	ublas::matrix<double> talks_to,
	ublas::matrix<double>& neighbor_list
)
{
	size_t n1 = A.size1();
	size_t n2 = A.size2();
	size_t n3 = nan_list.size1();
	if (n3 != 0)
	{
		size_t nan_count = nan_list.size1();
		size_t talk_count = talks_to.size1();

		ublas::matrix<double> nn(nan_count * talk_count, 2, 0);
		//可能有误
		for (size_t i = 0; i < talk_count; ++i)
		{
			for (size_t k = 0; k < nan_count; ++k)
			{
				nn(i * nan_count + k, 0) = nan_list(k, 1) + talks_to(i, 0);
				nn(i * nan_count + k, 1) = nan_list(k, 2) + talks_to(i, 1);
			}
		}

		//drop those nodes which fall outside the bounds of the original array
		double t1 = 0;
		for (size_t i = 0; i < nn.size1(); i++)
		{
			if (nn(i, 0) < 1 || nn(i, 0) > n1 || nn(i, 1) < 1 || nn(i, 1) > n2)t1 = t1 + 1;
		}

		ublas::matrix<double> L(nn.size1() - t1, 2, 0);
		double count = 0;
		for (size_t i = 0; i < nn.size1(); i++) //删掉没有在范围内的点,得到更新后的nn,也就是L
		{
			if (nn(i, 0) < 1 || nn(i, 0) > n1 || nn(i, 1) < 1 || nn(i, 1) > n2) {}
			else
			{
				L(count, 0) = nn(i, 0);
				L(count, 1) = nn(i, 1);
				count = count + 1;
			}
		}

		ublas::matrix<double> neighbor_list0(L.size1(), 3);
		for (size_t i = 0; i < L.size1(); i++)
		{
			for (size_t j = 0; j < 3; j++)
			{
				if (j > 0) {
					neighbor_list0(i, j) = L(i, j - 1);
				}
				else {
					neighbor_list0(i, j) = (L(i, 1) - 1) *L.size1() + L(i, 0);
				}
			}
		}

		//把ublas vector 转成 std vector做了，因为前者好像不能用unique这些函数
		// 两者之间函数可以封成函数，但目前没有做
		//把neighbor_list转换
		vector<vector<int> > v1(neighbor_list0.size1());   //行
		for (int i = 0; i < v1.size(); i++) {
			v1[i].resize(neighbor_list0.size2());          //列
			for (int j = 0; j < v1[i].size(); j++) {
				v1[i][j] = neighbor_list0(i, j);           //赋值
			}
		}
		sort(v1.begin(), v1.end());
		v1.erase(unique(v1.begin(), v1.end()), v1.end());   // unique那一步

		//把nan_list转换
		vector<vector<int> > v2(nan_list.size1());   //行
		for (int i = 0; i < v2.size(); i++) {
			v2[i].resize(nan_list.size2());          //列
			for (int j = 0; j < v2[i].size(); j++) {
				v2[i][j] = nan_list(i, j);           //赋值
			}
		}
		sort(v2.begin(), v2.end());
		v2.erase(unique(v2.begin(), v2.end()), v2.end());


		//比较二者差异 
		vector<vector<int> >diff;

		std::set_difference(v1.begin(), v1.end(), v2.begin(), v2.end(), std::inserter(diff, diff.begin()));

		//把得到的结果再转回到ublas vector
		ublas::matrix<double> trans(diff.size(), diff[0].size());
		for (size_t i = 0; i < diff.size(); i++)
		{
			for (int j = 0; j < diff[i].size(); j++) {
				trans(i, j) = diff[i][j];           //赋值
			}
		}
		ublas::matrix<double> neighbor_list = trans;
	}
	else {
		ublas::vector<double> neighbor_list(0);
	}
}

void inpaint_nans(
	ublas::matrix<double> A,
	int k,
	ublas::matrix<double>& B
)
{
	size_t n1 = A.size1();
	size_t n2 = A.size2();

	vector<vector<double> > nan_list;
	vector<vector<double> > known_list;
	double k1 = 0;
	double k2 = 0;
	for (size_t j = 0; j < n2; j++)
	{
		for (size_t i = 0; i < n1; i++)
		{

			//if (std::isnan(A(i, j)))
			if (std::isnan(A(i, j)) || std::isinf(A(i, j)))
			{
				nan_list.push_back(vector<double>(3));
				nan_list[k1][0] = j * n1 + i;
				nan_list[k1][1] = i;
				nan_list[k1][2] = j;
				k1 = k1 + 1;
			}
			else
			{
				known_list.push_back(vector<double>(2));
				known_list[k2][0] = j * n1 + i;
				known_list[k2][1] = A(i, j);
				k2 = k2 + 1;
			}
		}
	}
	B.resize(A.size1(), A.size2());
	B=A;
	if(nan_list.size()==0){}
	else{
	//把 标准vector转化成 ublas
	ublas::matrix<double> nan_list1(nan_list.size(), 3);
	for (int i = 0; i < nan_list.size(); i++)
	{
		for (int j = 0; j < 3; j++) {
			nan_list1(i, j) = nan_list[i][j];           //赋值
		}
	}

	ublas::matrix<double> talks_to(12, 2);
	talks_to <<= -2, 0,
		-1, -1,
		-1, 0,
		-1, 1,
		0, -2,
		0, -1,
		0, 1,
		0, 2,
		1, -1,
		1, 0,
		1, 1,
		2, 0;
    ublas::matrix<double> neighbor_list;
	identify_neighbors(A, nan_list1, talks_to, neighbor_list);

	//可用subrange优化
	ublas::matrix<double> all_list(nan_list1.size1() + neighbor_list.size1(), 3);
	for (size_t i = 0; i < all_list.size1(); i++)
	{
		for (size_t j = 0; j < 3; j++)
		{
			if (i < nan_list1.size1())
			{
				all_list(i, j) = nan_list1(i, j);
			}
			else
			{
				all_list(i, j) = neighbor_list(i - nan_list1.size1(), j);
			}
		}
	}

	vector<double> L1, L2, L3, L4;
	for (size_t i = 0; i < all_list.size1(); i++)
	{
		if (all_list(i, 1) >= 3 && all_list(i, 1) <= n1-2 && all_list(i, 2) >= 3 && all_list(i, 2) <= n2 - 2)
		{
			L1.push_back(i);
		}
		if ((((all_list(i, 1) == 2) || (all_list(i, 1) == (n1 - 1))) && (all_list(i, 2) >= 2) && (all_list(i, 2) <= (n2 - 1))) || (((all_list(i, 2) == 2) || (all_list(i, 2) == (n2 - 1))) && (all_list(i, 1) >= 2) && (all_list(i, 1) <= (n1 - 1))))
		{
			L2.push_back(i);
		}
		if (((all_list(i, 1) == 1) || (all_list(i, 1) == n1)) && (all_list(i, 2) >= 2) && (all_list(i, 2) <= (n2 - 1)))
		{
			L3.push_back(i);
		}
		if (((all_list(i, 2) == 1) || (all_list(i, 2) == n2)) && (all_list(i, 1) >= 2) && (all_list(i, 1) <= (n1 - 1)))
		{
			L4.push_back(i);
		}
	}

	ublas::compressed_matrix<double> fda(n1 * n2, n1 * n2);
	if (L1.size() > 0)
	{
		ublas::vector<double> T1(13);
		T1 <<= -2 * n1, -n1 - 1, -n1, -n1 + 1, -2, -1, 0, 1, 2, n1 - 1, n1, n1 + 1, 2 * n1;
		ublas::vector<double> T2(13);
		T2 <<= 1, 2, -8, 2, 1, -8, 20, -8, 1, 2, -8, 2, 1;
		for (size_t i = 0; i < L1.size(); i++)
		{
			for (size_t j = 0; j < 13; j++)
			{
				fda(all_list(L1[i]), all_list(L1[i]) + T1(j)) = T2(j);
			}
		}
	}
	else
	{
		ublas::compressed_matrix<double> fda(n1 * n2, n1 * n2, all_list.size1() * 5);
	}

	if (L2.size() > 0)
	{
		ublas::vector<double> T1(5);
		T1 <<= -n1, -1, 0, 1, n1;
		ublas::vector<double> T2(5);
		T2 <<= 1, 1, -4, 1, 1;
		for (size_t i = 0; i < L2.size(); i++)
		{
			for (size_t j = 0; j < 5; j++)
			{
				fda(all_list(L2[i]), all_list(L2[i]) + T1(j)) += T2(j);
			}
		}
	}
	if (L3.size() > 0)
	{
		ublas::vector<double> T1(3);
		T1 <<= -n1, -0, n1;
		ublas::vector<double> T2(3);
		T2 <<= 1, -2, 1;
		for (size_t i = 0; i < L3.size(); i++)
		{
			for (size_t j = 0; i < 3; i++)
			{
				fda(all_list(L3[i]), all_list(L3[i]) + T1(j)) += T2(j);
			}
		}
	}
	if (L4.size() > 0)
	{
		ublas::vector<double> T1(3);
		T1 <<= -1, -0, 1;
		ublas::vector<double> T2(3);
		T2 <<= 1, -2, 1;
		for (size_t i = 0; i < L4.size(); i++)
		{
			for (size_t j = 0; i < 3; i++)
			{
				fda(all_list(L4[i]), all_list(L4[i]) + T1(j)) += T2(j);
			}
		}
	}



	ublas::matrix<double> fdaknown(fda.size1(), known_list.size());
	for (size_t i = 0; i < fdaknown.size1(); i++)
	{
		for (size_t j = 0; j < fdaknown.size2(); j++) {
			fdaknown(i, j) = fda(i, known_list[j][0]);
		}
	}

	ublas::vector<double> Aknown(known_list.size());
	for (size_t i = 0; i < Aknown.size(); i++)Aknown(i) = known_list[i][1];

	ublas::vector<double> rhs = -prod(fdaknown, Aknown);


	// //找到非零行
	vector<double> k0;
	for (size_t i = 0; i < n1 * n2; i++)
	{
		for (size_t j = 0; j < nan_list1.size1(); j++)
		{
			if (fda(i, nan_list1(j, 0)) == 0)
			{
				k0.push_back(i);
				break;
			}
		}
	}

	// cout << k0.size() << std::endl;
	//cout << nan_list.size() << std::endl;

	Eigen::MatrixXd A0(k0.size(),nan_list1.size1());
	for (size_t i = 0; i < k0.size(); i++)
	{
		for (size_t j = 0; j < nan_list1.size1(); j++)
		{
			A0(i,j)=1/i+1/j;
			(k0[i],nan_list1(j,0)); 
		}
		
	}
	Eigen::VectorXd rhs0(k0.size());
	for (size_t i = 0; i < k0.size(); i++)
	{
		rhs0(i)=i;
		//rhs(k0[i]);
	}


	Eigen::VectorXd b2(nan_list.size());
	b2=pseudoInverse(A0)*rhs0;


	for (size_t i = 0; i < nan_list1.size1(); i++)
	{
		B(nan_list1(i,0))=b2(i);
	}
	//cout << A << std::endl;
	
	}
}


void OptionPrice(
	ublas::vector<double>x,
	ublas::vector<double>z,
	ublas::vector<double>L,
	ublas::vector<double>V_initial,
	ublas::matrix<double>psi,
	double T1,
	double T2,
	size_t nt,
	double r,
	double q,
	double y0,
	int call_put,//call:1, put:-1
	ublas::vector<double>& V					//return
)
{
	// if (call_put != 1 || call_put != -1) throw"error of option type, call_put must set to 1(call), or -1(put).";

	//M为稀疏矩阵
	size_t nx = x.size();
	size_t nz = z.size();
	double x_max = x[nx - 1], x_min = x[0];
	double z_max = z[nz - 1], z_min = z[0];
	double dx = (x_max - x_min) / (nx - 1);
	double dt = (T2 - T1) / (nt - 1);
	double dz = (z_max - z_min) / (nz - 1);
    
	ublas::vector<double>z1(z.size());
	fnexp(z,z1);
	ublas::vector<double>z2(z.size(),1);
	ublas::vector<double>C(z.size());
    ublas::vector<double>kb1=prod(psi,z1);
	ublas::vector<double>kb2=prod(psi,z2);
	C = y0 * element_div( kb1* dz, kb2* dz);
	
    //这里本不应注释
	// ublas::matrix<double> C1(C.size(),1);
	// for (size_t i = 0; i < C1.size1(); i++)C1(i,0)=C(i);
	// //std::cout<< C1<<std::endl;
	// inpaint_nans(C1, 3, C1); 

	// ublas::vector<double> C2(C1.size1());
	// for (size_t i = 0; i < C2.size(); i++)C2(i)=C1(i,0);
	// C=C2;		

	for (size_t i = 0; i < C.size(); i++)
	{
			if (C(i)< 0)C(i) = 0;
	}

	ublas::vector<double> A(nx - 2), A_plus(nx - 2), A_minus(nx - 2);
	for (size_t i = 0; i < A.size(); ++i)
	{
		A(i) = 1 + L(i + 1) * L(i + 1) * C(i + 1) * dt / dx / dx + dt * q;
		A_plus(i) = (r - q + 0.5 * L(i + 1) * L(i + 1) * C(i + 1)) * dt / 2 / dx - 0.5 * L(i + 1) * L(i + 1) * C(i + 1) * dt / dx / dx;
		A_minus(i) = -(r - q + 0.5 * L(i + 1) * L(i + 1) * C(i + 1)) * dt / 2 / dx - 0.5 * L(i + 1) * L(i + 1) * C(i + 1) * dt / dx / dx;
	}



	//以三对角矩阵(banded_matrix)存储M
	//ublas::banded_matrix<double> M(nx, nx, 1, 1);//size1,size2,lower,upper; Allocates an uninitialized banded_matrix that holds (lower + 1 + upper) diagonals
	ublas::matrix<double> M(nx, nx,0);
	M(0, 0) = 1.0; M(0, 1) = -1.0;
	M(nx - 1, nx - 1) = 1.0; M(nx - 1, nx - 2) = -1.0;


	for (size_t i = 1; i < nx - 1; ++i)
	{
		for (size_t j = i - 1; j <= i + 1; ++j)
		{
			if (j == i - 1) M(i, j) = A_minus(i - 1);      //upper, (i,j)=(1,0)处存储的是A_minus(0)
			else if (j == i)M(i, j) = A(i - 1);            //mid,   (i,j)=(1,1)处存储的是A(0)
			else if (j == i + 1) M(i, j) = A_plus(i - 1);  //lower, (i,j)=(1,2)处存储的是A_plus(0)
		}
	}

	V = V_initial;
	for (size_t t = 1; t < nt; ++t)
	{
		//ublas::vector<double> f = V;

		if (call_put == 1)
		{
			V(0) = exp(x(0)) * exp(-r * (T1 + t * dt)) * dx + 1.0 / 2.0 * exp(x(0)) * exp(-r * (T1 + t * dt)) * pow(dx, 2) + 1 / 6 * exp(x(0)) * exp(-r * (T1 + t * dt)) * pow(dx, 3) +
				1.0 / 24.0 * exp(x(0)) * exp(-r * (T1 + t * dt)) * pow(dx, 4) + 1.0 / 120.0 * exp(x(0)) * exp(-r * (T1 + t * dt)) * pow(dx, 5);
			V(V.size() - 1) = 0.0;
		}
		else if (call_put == -1)
		{
			V(0) = 0.0;
			V(V.size() - 1) = exp(x(nx - 1)) * exp(-r * (T1 + t * dt)) * dx + 1.0 / 2.0 * exp(x(nx - 1)) * exp(-r * (T1 + t * dt)) * pow(dx, 2) + 1 / 6 * exp(x(nx - 1)) * exp(-r * (T1 + t * dt)) * pow(dx, 3) +
				1.0 / 24.0 * exp(x(nx - 1)) * exp(-r * (T1 + t * dt)) * pow(dx, 4) + 1.0 / 120.0 * exp(x(nx - 1)) * exp(-r * (T1 + t * dt)) * pow(dx, 5);
		}


		//Matlab: V = M\f
		//===============LU分解求解线性方程组======================
		//link1: https://stackoverflow.com/questions/1225411/boosts-linear-algebra-solution-for-y-ax(关于ublas中如何使用LU分解)
		//link2：https://stackoverflow.com/questions/26404106/what-does-lu-factorize-return(关于LU分解)

		ublas::permutation_matrix<size_t> pm(M.size1());//列主元的高斯消元法
		int res = lu_factorize(M, pm);//LU分解并将L(对角线为1)保存在M左下角；U(对角线不为1)保存在M右上角
		if (res != 0)throw"error in LV decomposition.";
		else {
			lu_substitute(M, pm, V);//求解线性方程组,MV=f，并将结果保存在f中
		}
	// 	//===============LU分解求解线性方程组======================
	    for (size_t i = 0; i < V.size(); i++)
		{
			if(V(i)<0)V(i)=0;
		}
		
	// 	// for (auto iter = V.begin(); iter < V.end(); ++iter)
	// 	// 	if (*iter < 0)*iter = 0.0;
	}
}

void finite_difference_mat_b2(
	ublas::vector<double> x,
	ublas::vector<double> z,
	hestonParas hp,
	ublas::matrix<double>& T_x1,
	ublas::matrix<double>& T_x2,
	ublas::matrix<double>& T_z1,
	ublas::matrix<double>& T_z2,
	ublas::matrix<double>& T_zb1,
	ublas::matrix<double>& T_zb2
)
{
	size_t nx1 = x.size();
	size_t nz1 = z.size();

	ublas::vector<double> dx = subrange(x, 1, nx1) - subrange(x, 0, nx1 - 1);
	ublas::vector<double> dx_i_minus = subrange(dx, 0, dx.size() - 1);
	ublas::vector<double> dx_i = subrange(dx, 1, dx.size());

	ublas::vector<double> c_x_minus = -element_div(dx_i, element_prod(dx_i_minus, dx_i_minus + dx_i));
	ublas::vector<double> c_x = element_div(dx_i - dx_i_minus, element_prod(dx_i_minus, dx_i));
	ublas::vector<double> c_x_plus = element_div(dx_i_minus, element_prod(dx_i, dx_i_minus + dx_i));
	c_x(0) = c_x(0) + c_x_minus(0);
	c_x(c_x.size() - 1) = c_x(c_x.size() - 1) + c_x_plus(c_x_plus.size() - 1);

	ublas::vector<double> one1(dx_i_minus.size(), 2);
	ublas::vector<double> x_x_minus = element_div(one1, element_prod(dx_i_minus, dx_i_minus + dx_i));
	ublas::vector<double> x_x = -element_div(one1, element_prod(dx_i_minus, dx_i));
	ublas::vector<double> x_x_plus = element_div(one1, element_prod(dx_i, dx_i_minus + dx_i));
	x_x(0) = x_x(0) + x_x_minus(0);
	x_x(x_x.size() - 1) = x_x(x_x.size() - 1) + x_x_plus(x_x_plus.size() - 1);

	//这段还未检查
	T_x1.resize(nx1 - 2, nx1 - 2);
	T_x2.resize(nx1 - 2, nx1 - 2);
	
	for (size_t i = 1; i < nx1 - 3; ++i)
	{
		for (size_t j = i - 1; j <= i + 1; ++j)
		{
			if (j == i - 1) T_x1(i, j) = c_x_minus(i - 1), T_x2(i, j) = x_x_minus(i - 1);
			else if (j == i)T_x1(i, j) = c_x(i - 1), T_x2(i, j) = x_x(i - 1);
			else if (j == i + 1) T_x1(i, j) = c_x_plus(i - 1), T_x2(i, j) = x_x_plus(i - 1);
		}
	}


	ublas::vector<double> dz = subrange(z, 1, nz1) - subrange(z, 0, nz1 - 1);
	ublas::vector<double> dz_i_minus = subrange(dz, 0, dz.size() - 1);
	ublas::vector<double> dz_i = subrange(dz, 1, dz.size());

	ublas::vector<double> c_z_minus = -element_div(dz_i, element_prod(dz_i_minus, dz_i_minus + dz_i));
	ublas::vector<double> c_z = element_div(dz_i - dz_i_minus, element_prod(dz_i_minus, dz_i));
	ublas::vector<double> c_z_plus = element_div(dz_i_minus, element_prod(dz_i, dz_i_minus + dz_i));
	// here we give a boundary condition for pe^z
	ublas::vector<double> c_z_b = c_z;
	c_z_b(0) = c_z_b(0) + (1 + dz_i_minus(0) - dz_i_minus(0) * 2 * hp.kappa * hp.theta / pow(hp.VoV, 2)) * c_z_minus(0);
	//
	// c_z(0) = c_z(0) + c_z_minus(0);
	// c_z(c_z.size() - 1) = c_z(c_z.size() - 1) + c_z_plus(c_z_plus.size() - 1);
	c_z=c_z_b;

	ublas::vector<double> one0(dz_i_minus.size(), 2);
	ublas::vector<double> x_z_minus = element_div(one0, element_prod(dz_i_minus, dz_i_minus + dz_i));
	ublas::vector<double> x_z = -element_div(one0, element_prod(dz_i_minus, dz_i));
	ublas::vector<double> x_z_plus = element_div(one0, element_prod(dz_i, dz_i_minus + dz_i));
	// here we give a boundary condition for pe^-z from zero flux condition
	ublas::vector<double> x_z_b = x_z;
	x_z_b(0) = x_z_b(0) + (1 + dz_i_minus(0) - dz_i_minus(0) * 2 * hp.kappa * hp.theta / pow(hp.VoV, 2)) * x_z_minus(0);
	//
	// x_z(0) = x_z(0) + x_z_minus(0);
	// x_z(x_z.size() - 1) = x_z(x_z.size() - 1) + x_z_plus(x_z_plus.size() - 1);
	x_z=x_z_b;


	// //这里还没有检查？
	
	T_z1.resize(nz1 - 2, nz1 - 2);
	T_z2.resize(nz1 - 2, nz1 - 2);
    T_zb1.resize(nz1 - 2, nz1 - 2);
	T_zb2.resize(nz1 - 2, nz1 - 2);
	for (size_t i = 1; i < nz1 - 3; ++i)
	{
		for (size_t j = i - 1; j <= i + 1; ++j)
		{
			if (j == i - 1){
				T_z1(i, j) = c_z_minus(i - 1);
				T_z2(i, j) = x_z_minus(i - 1);
				T_zb1(i, j) = c_z_minus(i - 1);
				T_zb2(i, j) = x_z_minus(i - 1);
			} 
			else if (j == i){
				T_z1(i, j) = c_z(i - 1);
				T_z2(i, j) = x_z(i - 1);
				T_zb1(i, j) = c_z_b(i - 1);
				T_zb2(i, j) = x_z_b(i - 1);
			}
			else if (j == i + 1){
				T_z1(i, j) = c_z_plus(i - 1);
				T_z2(i, j) = x_z_plus(i - 1);
				T_zb1(i, j) = c_z_plus(i - 1);
				T_zb2(i, j) = x_z_plus(i - 1);
			} 
		}
	}
}

void adi_p_b(ublas::vector<double> x,
	ublas::vector<double> z,
	ublas::vector<double> sigma,
	ublas::matrix<double> initial_p,
	double t_start,
	double t_end,
	double dt,
	hestonParas& hp,
	double r,
	double q,
	double alpha,
	ublas::matrix<double>& psi
)
{
	size_t nt = floor((t_end - t_start) / dt + 1);
	ublas::vector<double> t(nt);
	for (size_t i = 0; i < nt; ++i)t(i) = t_start + i * dt;

	size_t nx2 = x.size();
	size_t nz2 = z.size();
	double dx = (x(nx2 - 1) - x(0)) / (nx2 - 1);
	double dz = (z(nz2 - 1) - z(0)) / (nz2 - 1);

	ublas::matrix<double> T_S1, T_S2, T_z1, T_z2, T_zb1, T_zb2;
	finite_difference_mat_b2(x, z, hp, T_S1, T_S2, T_z1, T_z2, T_zb1, T_zb2);

	ublas::matrix<double> p = subrange(initial_p, 1, nx2 - 1, 1, nz2 - 1);//这里或许可以改进用切片

	ublas::vector<double> x_inner(nx2 - 2);//可以改进用切片
	ublas::vector<double> z_inner(nz2 - 2);
	ublas::vector<double> sigma_inner(sigma.size() - 2);
	for (size_t i = 0; i < x_inner.size(); ++i)x_inner(i) = x(i + 1);
	for (size_t i = 0; i < z_inner.size(); ++i)z_inner(i) = z(i + 1);
	for (size_t i = 0; i < sigma_inner.size(); ++i)sigma_inner(i) = sigma(i + 1);

	psi.resize(nx2, nz2);
	for (size_t i = 0; i <nx2; i++)
	{
		for (size_t j = 0; j < nz2; j++)
		{
			psi(i,j)=0;
		}
		
	}


	for (size_t i = 0; i < nt - 1; ++i)//共nt-1次循环
	{
		//std::cout << 5 << std::endl;
		ublas::matrix<double> sv;
		stackvector(sigma_inner, 1, sv);
		ublas::matrix<double> f0p1=element_prod(sv, T_S1);
		ublas::matrix<double> f0p2=prod(f0p1,p);
		ublas::matrix<double> F0p = hp.VoV * hp.rho * prod(f0p2, T_z1); 
		// ublas::matrix<double> F0p = hp.VoV * hp.rho *p;


		ublas::vector<double> temp(z_inner.size());
		for (size_t j = 0; j < z_inner.size(); ++j)temp(j) = exp(-z_inner(j)) / hp.Y0;
		stackvector(temp, 2, sv);
		ublas::matrix<double> F1 = -(hp.kappa * hp.theta - 0.5 * pow(hp.VoV,2)) * element_prod(T_zb1, sv) + hp.kappa * T_z1 + 0.5 * pow(hp.VoV,2)* element_prod(T_zb2, sv);
		//这里如果添加上后两项会变得不ok
	


		temp.resize(sigma_inner.size());
		for (size_t j = 0; j < sigma_inner.size(); ++j)temp(j) = sigma_inner(j) * sigma_inner(j);
		stackvector(temp, 1, sv);

		ublas::matrix<double> temp1(z_inner.size(),z_inner.size());
		for (size_t i = 0; i < temp1.size1(); i++)
		{			
			temp1(i,i) = exp(z_inner(i)) * hp.Y0;
		}
        ublas::matrix<double> f2p1=element_prod(sv, T_S1);
		ublas::matrix<double> f2p2=prod(f2p1, p);
		ublas::matrix<double> f2p3=element_prod(sv, T_S2);
		ublas::matrix<double> f2p4=prod(f2p3, p);
		ublas::matrix<double> F2p = (r - q) * prod(T_S1, p) + 0.5 * prod(f2p2, temp1) + 0.5 * prod(f2p4, temp1);


		ublas::matrix<double> A = p + dt * (F0p + prod(p, F1) + F2p);

		ublas::matrix<double> B1 = (A - alpha * dt * prod(p, F1));
		ublas::matrix<double> B2(nz2 - 2, nz2 - 2);
		for (size_t i = 0; i < nz2 - 2; i++)B2(i, i) = 1;
		ublas::matrix<double> B3 = B2 - alpha * dt * F1;

		ublas::permutation_matrix<size_t> pm2(B3.size1());

		int res3 = lu_factorize(B3, pm2);
		if (res3 != 0)throw"error in LV decomposition.";
		ublas::matrix<double> B3inv = ublas::identity_matrix<double>(B3.size1());
		lu_substitute(B3, pm2, B3inv);
		ublas::matrix<double> B = prod(B1, B3inv);

		// // std::cout << B.size1() <<std::endl;
		//ublas::matrix<double> B=A;

		ublas::compressed_matrix<double>I1(nx2 - 2, nx2 - 2);
		for (size_t i = 0; i < nx2 - 2; i++)I1(i, i) = 1;

		ublas::compressed_matrix<double>K1((nx2 - 2) * (nz2 - 2), (nx2 - 2) * (nz2 - 2), 0);
		ublas::compressed_matrix<double>K2((nx2 - 2) * (nz2 - 2), (nx2 - 2) * (nz2 - 2), 0);		
		ublas::compressed_matrix<double>K4((nx2 - 2) * (nz2 - 2), (nx2 - 2) * (nz2 - 2), 0);		
		ublas::compressed_matrix<double>K3((nx2 - 2) * (nz2 - 2), (nx2 - 2) * (nz2 - 2), 0);
		for (size_t i = 0; i < nz2 - 2; i++)
		{
			subrange(K1, i * (nx2 - 2), (i + 1) * (nx2 - 2) , i * (nx2 - 2), (i + 1) * (nx2 - 2) ) = T_S1;
			subrange(K2, i * (nx2 - 2), (i + 1) * (nx2 - 2) , i * (nx2 - 2), (i + 1) * (nx2 - 2)) = f2p1;
			subrange(K4, i * (nx2 - 2), (i + 1) * (nx2 - 2) , i * (nx2 - 2), (i + 1) * (nx2 - 2)) = f2p3;
			subrange(K3, i * (nx2 - 2), (i + 1) * (nx2 - 2) , i * (nx2 - 2), (i + 1) * (nx2 - 2)) = exp(z_inner(i)) * hp.Y0 * I1;
		}

        ublas::compressed_matrix<double>I2((nx2 - 2)*(nz2-2), (nx2 - 2)*(nz2-2),0);
		for (size_t i = 0; i < (nx2 - 2)*(nz2-2); i++)I2(i, i) = 1;
		ublas::compressed_matrix<double> Cnew((nx2 - 2)*(nz2 - 2), (nx2 - 2)*(nz2 - 2),0);
		Cnew= I2-alpha*dt*(-(r-q)*K1+0.5*prod(K3,K2)+0.5*prod(K3,K4));
		//Cnew=I2;



        ublas::vector<double> resh((nx2-2)*(nz2-2));
		for (size_t i = 0; i < nx2-2; i++)
		{
			for (size_t j = 0; j < nz2-2; j++)
			{
				resh(j*(nx2-2)+i)=B(i,j)-alpha*dt*F2p(i,j);
			}		
		}
		ublas::permutation_matrix<size_t> pm1(Cnew.size1());
		int res2 = lu_factorize(Cnew, pm1);
		lu_substitute(Cnew, pm1, resh);


		ublas::matrix<double> p(nx2 - 2, nz2 - 2);
		for (size_t i = 0; i < nx2-2; i++)
		{
			for (size_t j = 0; j < nz2-2; j++)
			{
				if(resh(j*(nx2-2)+i)<0)p(i,j)=0;
				//else p(i,j)=resh(j*(nx2-2)+i);
				p(i,j)=1;
			}			
		}
	}
	subrange(psi, 1, nx2 - 1, 1, nz2 - 1) = p;
}

void meshgrid(ublas::vector<double>& x, ublas::vector<double>& y, ublas::matrix<double>& X, ublas::matrix<double>& Y)
{
	/*
	* X : x作为每一行，共y.size()行
	* Y ：y作为每一列, 共x.size()列
	*
	* matlab:
	[A,B]=Meshgrid(a,b)
	生成size(b)Xsize(a)大小的矩阵A和B。它相当于a从一行重复增加到size(b)行，把b转置成一列再重复增加到size(a)列
	*/
	size_t nx = x.size();
	size_t ny = y.size();

	X.resize(ny, nx);
	Y.resize(ny, nx);

	for (size_t i = 0; i < ny; ++i)
		for (size_t j = 0; j < nx; ++j)
		{
			X(i, j) = x[j];
			Y(i, j) = y[i];
		}

}

static double Mtools_T[] = {
 9.60497373987051638749E0, 9.00260197203842689217E1, 2.23200534594684319226E3,
 7.00332514112805075473E3, 5.55923013010394962768E4 };
static double Mtools_U[] = {/* 1.00000000000000000000E0,*/
 3.35617141647503099647E1, 5.21357949780152679795E2, 4.59432382970980127987E3,
 2.26290000613890934246E4, 4.92673942608635921086E4 };
static  double Mtools_nep[] = {
 2.46196981473530512524E-10, 5.64189564831068821977E-1, 7.46321056442269912687E0,
 4.86371970985681366614E1,   1.96520832956077098242E2,  5.26445194995477358631E2,
 9.34528527171957607540E2,   1.02755188689515710272E3,  5.57535335369399327526E2 };
static  double Mtools_neq[] = {/* 1.00000000000000000000E0,*/
 1.32281951154744992508E1, 8.67072140885989742329E1, 3.54937778887819891062E2,
 9.75708501743205489753E2, 1.82390916687909736289E3, 2.24633760818710981792E3,
 1.65666309194161350182E3, 5.57535340817727675546E2 };
static  double Mtools_ner[] = {
 5.64189583547755073984E-1, 1.27536670759978104416E0, 5.01905042251180477414E0,
 6.16021097993053585195E0,  7.40974269950448939160E0, 2.97886665372100240670E0 };
static  double Mtools_nes[] = {/* 1.00000000000000000000E0,*/
 2.26052863220117276590E0, 9.39603524938001434673E0, 1.20489539808096656605E1,
 1.70814450747565897222E1, 9.60896809063285878198E0, 3.36907645100081516050E0 };
double normal(double a)
{
	double x;
	double y;
	double z;

	x = a * DBA_SQRT2OVER2;
	z = (x > 0 ? x : -x);
	if (z < DBA_SQRT2OVER2)
		y = 0.5 + 0.5 * normal_error(x);
	else {
		y = 0.5 * normal_error_comp(z);
		if (x > 0.0)
			y = 1.0 - y;
	}
	return(y);
}
inline
double polynomial_1(double x, double coef[], int N)
{
	double* p = coef;
	double ans = x + *p++;
	int i = N - 1;
	do ans = ans * x + *p++;
	while (--i);
	return(ans);
}
inline
double polynomial(double x, double coef[], int N)
{
	double* p = coef;
	double ans = *p++;
	do ans = ans * x + *p++;
	while (--N);
	return(ans);
}
double normal_error(double x)
{
	double y;
	double z;

	z = x * x;
	if (z > 1.0)
		return(1.0 - normal_error_comp(x));
	y = x * polynomial(z, Mtools_T, 4) / polynomial_1(z, Mtools_U, 5);
	return(y);
}
double normal_error_comp(double a)
{
	double p;
	double q;
	double x;
	double y;
	double z;

	x = (a > 0.0 ? a : -a);
	if (x < 1.0)
		return(1.0 - normal_error(a));
	z = -a * a;
	if (z < -DBA_MAXLOG) {
	under:
		dba_error("normal_error_comp", DBA_ERROR_UNDERFLOW);
		return(0.0);
	}
	z = exp(z);
	if (x < 8.0) {
		p = polynomial(x, Mtools_nep, 8);
		q = polynomial_1(x, Mtools_neq, 8);
	}
	else {
		p = polynomial(x, Mtools_ner, 5);
		q = polynomial_1(x, Mtools_nes, 6);
	}
	y = (z * p) / q;
	if (a < 0.0)
		y = 2.0 - y;
	if (y == 0.0)
		goto under;
	return(y);
}



double European_Vega(double spot, double strike, double maturity, double risk_free_rate, double dividend_rate, double vol)
{
	// returns vega.  call and put have the same vega.

	double d1, answer;
    double PI = boost::math::constants::pi<double>();

	d1 = (log(spot / strike) + (risk_free_rate - dividend_rate + vol * vol / 2.) * maturity) /
		vol / sqrt(maturity);
	answer = spot * sqrt(maturity) * exp(-dividend_rate * maturity) * exp(-d1 * d1 / 2.) / sqrt(2. * PI);

	return answer;
}

double European_BS(long call_put_flag, double spot, double strike, double maturity, double risk_free_rate, double dividend_rate, double vol)
{    
	//这里有问题
	double EPS=1e-8;
	if (vol < EPS)
		vol = EPS;
	if (maturity < EPS)
		maturity = EPS;
	if (strike < EPS)
		strike = EPS;

	double fac = vol * sqrt(maturity);
	double d1 = (log(spot / strike) + (risk_free_rate - dividend_rate) * maturity) / fac + fac * 0.5;
	double d2 = d1 - fac;
	double answer = call_put_flag * (spot * exp(-dividend_rate * maturity) * normal(call_put_flag * d1) -
		strike * exp(-risk_free_rate * maturity) * normal(call_put_flag * d2));

	return answer;
}

double ImpliedVol(long call_put_flag, double option_price, double spot, double strike, double maturity, double risk_free_rate, double dividend_yield)
{   
	double EPS=1e-8;
	double price_guess, vega, error;
	double imp_vol = 0.5;  // 1st guess
	const long max_iteration = 200;
	long i = 0;
	double precision = EPS * 1.e-4;

	if (option_price <= precision)
		throw("option_price <= 0, can not imply vol !!");

	do
	{
		price_guess = European_BS(call_put_flag, spot, strike, maturity, risk_free_rate,
			dividend_yield, imp_vol);
		vega = European_Vega(spot, strike, maturity, risk_free_rate, dividend_yield,
			imp_vol);
		//printf("K=%f,vega=%e\b", strike, vega);
		if (vega < precision)
			throw("vega is near zero, difficult to iterate !!");

		error = price_guess - option_price;
		imp_vol -= error / vega;

		i++;
		if (i > max_iteration)
			throw("Number of iterations exceeds maximum allowed !!");
	} while (fabs(error) > precision);

	return imp_vol;

}


void calibration_FinitePoint4(
	ublas::vector<double>x,				//x,z,Maturity : original column vector
	ublas::vector<double>z,
	ublas::vector<double>Maturity,        //model maturities
	ublas::matrix<double>L_initial,		//initial value of local term
	ublas::matrix<double>impvs,			//market implied volsurface
	ublas::vector<double>strikes,         //market strikes
	ublas::vector<double>maturities,      //market maturities
	hestonParas& hp,              //heston paramaters
	double r,                               //risk free rate
	double q,                               //dividend 
	double alpha,                           //scheme parameters?
	double epsilon,                         //
	double epsilon2,                        //
	double step_size,                       //which step size?
	int NN,                                //

	ublas::matrix<double>& Leverage_surf,   //return: 
	ublas::matrix<double>& impv_surf,       //return: 
	ublas::vector<double>& x_L				//return:
)
{
	x_L.resize(x.size());
	x_L=x;
	size_t xn1 = x.size();
	size_t zn1 = z.size();
	double x_max = x[x.size()- 1], x_min = x[0];
	double z_max = z[z.size() - 1], z_min = z[0];
	double dx = (x_max - x_min) / (x.size() - 1);
	double dz = (z_max - z_min) / (z.size() - 1);

	////按照原code,先产生完整x_dense,x_sparse,再删除最后19个；
	//size_t dense_para_x = 20;
	//ublas::vector<double> x_dense(xn * dense_para_x);
	//ublas::vector<double> x_increased(xn * dense_para_x);
	//double dx_dense = dx / dense_para_x;
	//for (size_t i = 0; i < x.size(); ++i)
	//{
	//	for (size_t j = 0; j < dense_para_x; ++j)
	//	{
	//		x_dense(i * dense_para_x + j) = x[i];
	//		x_increased(i * dense_para_x + j) = dx_dense * j;
	//	}
	//}
	//ublas::vector<double> x_sparse = x_dense;
	//x_dense = x_dense + x_increased; //dense grids
	//x_dense = subrange(x_dense, 0, x_dense.size() - 19);//原code：删除最后dense_para_x-1=19个元素； 这里size =  原x_dense.size() - 19
	//x_sparse = subrange(x_sparse, 0, x_sparse.size() - 19);

	//更新code,产生完整x_dense,x_sparse,初始就少19个；
	size_t dense_para_x = 20;
	ublas::vector<double> x_dense((x.size()- 1) * dense_para_x + 1);
	ublas::vector<double> x_increased((x.size() - 1) * dense_para_x + 1);
	double dx_dense = dx / dense_para_x;
	for (size_t i = 0; i < x.size(); ++i)
	{
		for (size_t j = 0; j < dense_para_x && i * dense_para_x + j < x_dense.size(); ++j)
		{
			x_dense(i * dense_para_x + j) = x[i];
			x_increased(i * dense_para_x + j) = dx_dense * j;
		}
	}
	ublas::vector<double> x_sparse = x_dense;
	x_dense = x_dense + x_increased; //dense grids

	size_t dense_para_z = 1;
	double dz_dense = dz / dense_para_z;
	ublas::vector<double> z_dense(z.size() * dense_para_z);
	ublas::vector<double> z_increased(z.size() * dense_para_z);
	for (size_t i = 0; i < z.size(); ++i)
	{
		for (size_t j = 0; j < dense_para_z; ++j)
		{
			z_dense(i * dense_para_z + j) = z[i];
			z_increased(i * dense_para_z + j) = dz_dense * j;
		}
	}
	z_dense = z_dense + z_increased; //dense grids


	double sigma_initial = 1.0;
	double dt = 1e-5;
	ublas::matrix<double> initial_p;
	bisection_initial(x_dense, z_dense, hp, r, q, sigma_initial, dt, initial_p);

	double lb = -1, rb = 1;//boundary condition for L

	ublas::vector<double> V_initial_call(x_dense.size());
	ublas::vector<double> V_initial_put(x_dense.size());
	ublas::vector<double> V_call(x_dense.size(),0);
	ublas::vector<double> V_put(x_dense.size(),0);
	for (size_t i = 0; i < x_dense.size(); ++i)
	{
		V_initial_call(i) = std::max(1 - exp(x_dense(i)), 0.0);
		V_initial_put(i) = std::max(exp(x_dense(i)) - 1, 0.0);
	}

	ublas::vector<double> L = column(L_initial, 0);//L = L_initial(:, 1);


	Leverage_surf.resize(x.size(), Maturity.size());
	impv_surf.resize(x.size(), Maturity.size());


	double t_start;
	double t_end;
	for (size_t t = 0; t < Maturity.size()-1 ; ++t)
	{

		double t_start = (t == 0 ? 0.0 : Maturity(t));
		double t_end = (t == 0 ? Maturity(t) : Maturity(t + 1));

		ublas::matrix<double> v(x.size(),NN + 1);
		ublas::matrix<double> s(x.size(), NN + 1);

		double beta_1 = 0.9, beta_2 = 0.9;
		for (size_t n = 0; n < NN; n++)
		{
			//V_put;

			ublas::vector< double> impv(x.size(),0);
			ublas::vector< double> vega(x.size(),0);

			interp::cardinal_cubic_b_spline<double> spline(L.begin(), L.end(), x(0), x(1) - x(0));//L_dense=interp1(x,L,x_dense,'spline');
			ublas::vector< double> L_dense(x_dense.size());
			for (size_t k = 0; k < L_dense.size(); k++)L_dense(k) = spline(x_dense(k));
			ublas::vector< double> L_old = L;
			ublas::vector< double> G(L.size());//G = log(L);
			for (size_t k = 0; k < L.size(); k++)G(k) = log(L(k));

			

			if (t_end < dt)
			
			{
				std::cout << 0 << std::endl;
				bisection_initial(x_dense, z_dense, hp, r, q, sigma_initial, dt, initial_p);//会改变initial_p
				// std::cout << dt << std::endl;
				OptionPrice(x_dense, z_dense, L_dense, V_initial_call, initial_p, t_start, t_end, 5, r, q, hp.Y0, 1, V_call);//不改变initial_p，仅是输入
				OptionPrice(x_dense, z_dense, L_dense, V_initial_put, initial_p, t_start, t_end, 5, r, q, hp.Y0, -1, V_put);//不改变initial_p，仅是输入


			}
			else if (t_start<dt && t_end>dt)
			{
				bisection_initial(x_dense, z_dense, hp, r, q, sigma_initial, dt, initial_p);//会改变initial_p
				std::cout << 1 << std::endl;
				OptionPrice(x_dense, z_dense, L_dense, V_initial_call, initial_p, t_start, dt, 5, r, q, hp.Y0, 1, V_call);//不改变initial_p，仅是输入
				OptionPrice(x_dense, z_dense, L_dense, V_initial_put, initial_p, t_start, dt, 5, r, q, hp.Y0, -1, V_put);//不改变initial_p，仅是输入

				ublas::vector< double> V_middle_call = V_call;
				ublas::vector< double> V_middle_put = V_put;

				adi_p_b(x_dense, z_dense, L_dense, initial_p, dt, t_end, (t_end - dt) / 1, hp, r, q, alpha,initial_p);//intial_p既是输入也是输出

				OptionPrice(x_dense, z_dense, L_dense, V_middle_call, initial_p, dt, t_end, 5, r, q, hp.Y0, 1, V_call);//不改变initial_p，仅是输入
				OptionPrice(x_dense, z_dense, L_dense, V_middle_put, initial_p, dt, t_end, 5, r, q, hp.Y0, -1, V_put);//不改变initial_p，仅是输入


			}
			else
			{   
				std::cout << 2 << std::endl;
				//std::cout << t_end << std::endl;
				adi_p_b(x_dense, z_dense, L_dense, initial_p, t_start, t_end, (t_end - t_start) / 1, hp, r, q, alpha,initial_p);//intial_p既是输入也是输出
				OptionPrice(x_dense, z_dense, L_dense, V_initial_call, initial_p, t_start, t_end, 5, r, q, hp.Y0, 1, V_call);//不改变initial_p，仅是输入
				OptionPrice(x_dense, z_dense, L_dense, V_initial_put, initial_p, t_start, t_end, 5, r, q, hp.Y0, -1, V_put);//不改变initial_p，仅是输入
			}
        


			ublas::matrix<double> psi(initial_p.size1(), initial_p.size2());
			psi = initial_p * exp(-r * Maturity(t));

			ublas::vector<double> G_T(G.size(), 0);
			if (t > 1) {
				for (size_t i = 0; i < Leverage_surf.size1(); i++)
				{
					G_T(i) = (G(i) - log(Leverage_surf(i, t - 2))) / (t_end - t_start);
				}
			}

			ublas::vector<double>sign1(xn1, 0);

			for (size_t j = 0; j < x.size(); j++)
			{				
				ublas::vector<int>d(1,0);
				for (size_t i = 0; i < x_dense.size(); i++)
				{
					if(x_dense(i)==x(j)){
						d(0)=i;
						break;
					}
				}	
				if (x(j) > 0) 
				{
					if (V_call(d(0)) >= exp(-q * t_end)) {
						sign1(j) = 1; 
						impv(j) = NAN;
					}
					else {
						
						//impv(j) = ImpliedVol(1,V_call(d(0)),1, exp(x(j)), r, t_end, q);
	                    impv(j)=0.1;
					}
				}
				else {
					if (V_put(d(0)) >= exp(-r * t_end) * exp(x(j))) {
						sign1(j) = 1;//
						impv(j) = NAN;
					}
					else {
						//impv(j)=ImpliedVol(-1,1,V_call(d(0)) , exp(x(j)), r, t_end, q);
						impv(j)=0.1;
					}
				}

				if (isnan(impv(j))) {}
				else {
					//vega(j) = European_Vega(1, exp(x(j)), r, t_end, impv(j), q);
					vega(j)=0.1;
				}
				if (vega(j) == 0) {
					vega(j) = 1e-6;
				}
			}


			ublas::vector<double> gradient(x.size(),0);
			ublas::vector<double> int_psi(psi.size1(),0);
			for (size_t k = 0; k < strikes.size(); k++)
			{
				for (size_t h = 0; h < maturities.size(); h++)
				{
					if (impvs(k, h) == 0) {}
					else {
						ublas::vector<double> ep(z_dense.size());
						fnexp(z_dense,ep);
						int_psi = hp.Y0* dz_dense * prod(psi,ep);
						vector<double> int_psi0;
						for (size_t i = 0; i < x_dense.size(); i++)
						{
							if(x_dense(i)==x_sparse(i))int_psi0.push_back(int_psi(i));
						}

						ublas::vector<double> int_psi1(int_psi0.size());
						for (size_t i = 0; i < int_psi1.size(); i++)
						{
							int_psi1(i)=int_psi0[i];
						}

						ublas::vector<double>ones(x.size(),1);
						ublas::vector<double>x1(x.size());
						ublas::vector<double>x2(x.size());
                        fnexp(x,x1);
                        fnexp(x+2*G,x2);
						
						ublas::vector<double> f0 = element_prod(x1 - strikes(k)*ones, x1 - strikes(k)*ones);
						ublas::vector<double> f1 = (1e-6 + pow(t_end - maturities(h), 2)) * ones + f0;
						ublas::vector<double> f2 = element_prod(f1, f1);
						ublas::vector<double> f3 = element_prod(f2, f2);
						
						gradient += element_div(element_prod(element_div(element_prod(x2, impv - impvs(k, h) * ones), int_psi1), vega), f3);
					}
				}
			}			
			ublas::vector<double>x3(x.size()-2);
			fnexp(-2 * subrange(x, 1, x.size() - 1),x3);

			subrange(gradient, 1, gradient.size() - 1) = subrange(gradient, 1, gradient.size() - 1) - epsilon * element_prod(x3, (subrange(G, 2, G.size()) + subrange(G, 0, G.size() - 2) - 2 * subrange(G, 1, G.size() - 1)) / pow(dx, 2) - (subrange(G, 2, G.size()) - subrange(G, 0, G.size() - 2)) / 2 / dx) + epsilon2 * subrange(G_T, 1, G_T.size() - 1);
			
			for (size_t i = 0; i < gradient.size(); i++)
			{
				if (isnan(gradient(i))) {
					gradient(i) = 0;
				}
				else if (isinf(gradient(i))) {
					gradient(i) = 1; //这里不知
				}
			}
			gradient = gradient + 1e-2 * sign1;
			ublas::matrix<double> G1(gradient.size(),1);
			for (size_t i = 0; i < G1.size1(); i++)G1(i,0)=gradient(i);
			inpaint_nans(G1,3,G1);
			ublas::vector<double> G2(G1.size1());
			for (size_t i = 0; i < G2.size(); i++)G2(i)=G1(i,0);
			gradient=G2;		

			column(v, n+1) = beta_1 * column(v, n) + (1 - beta_1) * gradient;
			column(s, n+1) = beta_2 * column(s, n) + (1 - beta_1) * element_prod(gradient, gradient);
			ublas::vector<double>v_modefied(v.size1());
			ublas::vector<double>s_modefied(s.size1());
			ublas::vector<double>g_modefied(v.size1());
			v_modefied = column(v, n+1) / (1 - pow(beta_1, n + 1));
			s_modefied = column(s, n+1) / (1 - pow(beta_2, n + 1));
			ublas::vector<double> modefied1(s_modefied.size(), pow(10, -3));
			ublas::vector<double> modefied2 = s_modefied;
			for (size_t i = 0; i < modefied2.size(); i++)modefied2(i) = sqrt(modefied2(i));
			g_modefied = element_div(step_size * v_modefied, modefied1 + modefied2);
			G = G - g_modefied;
			G(0) = -lb * dx + G(1);
			G(G.size() - 1) = rb * dx + G(G.size() - 2);

			for (size_t i = 0; i < G.size(); i++)
			{
				if (G(i) < -10) {
					G(i) = -10;
				}
				if (G(i) > 3) {
					G(i) = 3;
				}
			}

			fnexp(G,L);
			double err_L = norm_2(L_old - L);

			ublas::vector<double>err_G = L_old-L;
			for (size_t i = 0; i < err_G.size(); i++)err_G(i) = abs(err_G(i));
			vector<double> S1;
			for (size_t i = 0; i < x.size(); i++)
			{
				if (exp(x(i))>=strikes(0)-0.5 && exp(x(i))<=strikes(strikes.size()-1)+0.5)
				{
				S1.push_back(err_G(i));
				}
			}		
			double err=*max_element(S1.begin(),S1.end());

			if (err < 1e-1) {
				printf("Maturity:%f, inter:%d, error_L:%f ,err:%f \n", t_end, n, err_L,err);
				column(Leverage_surf, t ) = L_old;
				column(impv_surf, t ) = impv;
				V_initial_call = V_call;
				V_initial_put = V_put;
				break;				
			}
			if (n == NN-1) {
				printf("Not converge");
				column(Leverage_surf, t) = L_old;
				column(impv_surf, t) = impv;
				V_initial_call = V_call;
				//initial_p=p;
				V_initial_put = V_put;
			}
			
		}

	}
	
}

void interp2(ublas::matrix<double>& X, ublas::matrix<double>& Z, ublas::matrix<double>& impv_surf, ublas::matrix<double>& S, ublas::matrix<double>& M, ublas::matrix<double>& impv_model)
{

	size_t n1 = S.size1();//行数
	size_t n2 = S.size2();//列数
	impv_model.resize(n1, n2);

	size_t m1 = X.size1();//行数
	size_t m2 = X.size2();//列数
	for (size_t i = 0; i < n1; ++i)
	{
		for (size_t j = 0; j < n2; ++j)
		{
			double S0 = S(i, j);//=S(0,j)
			double M0 = M(i, j);//=M(i,0)

			size_t k = 0, l = 0;
			for (k = 0; k < m2; ++k)
			{
				if (S0 < X(0, k))break; //X(0, k-1) <= S0 < X(0, k)
			}
			for (l = 0; l < m1; ++l)
			{
				if (M0 < Z(l, 0))break; //Z(0, l-1) <= M0 < Z(0, l)
			}

			if (k == 0) k = 1;
			if (k == m2) k = m2 - 1;
			if (l == 0) l = 1;
			if (l == m1) l = m1 - 1;




			impv_model(i, j) =
				(impv_surf(l - 1, k - 1) * (M0 - Z(l - 1, 0)) * (S0 - X(0, k - 1)) +
					impv_surf(l, k) * (Z(l, 0) - M0) * (X(0, k) - S0) +
					impv_surf(l, k - 1) * (Z(l, 0) - M0) * (S0 - X(0, k - 1)) +
					impv_surf(l - 1, k) * (M0 - Z(l - 1, 0)) * (X(0, k) - S0))
				/ (Z(l, 0) - Z(l - 1, 0)) / (X(0, k) - X(0, k - 1));
		}
	}

}
int main()
{
	hestonParas hp = { 0.059,2.0,0.484,-0.063,0.025 };//θ, κ，ξ(VoV), ρ, y0; 
	vector<double> strikes0 = { 0.7925,0.8087,0.8491,0.8895,0.93,0.9704,1.0108,1.0513,1.0917,1.1321,1.1726 };
	vector<double> maturities0 = { 0.0438,0.1342,0.274,0.5425 };
	vector<vector<double>> impvs0 = {
		{0.3742,	0.0000,	0.2373,	0.2150},
		{0.3536,	0.2668,	0.2259,	0.2128},
		{0.2942,	0.2312,	0.2078,	0.2027},
		{0.2415,	0.2072,	0.1927,	0.1938},
		{0.1912,	0.1881,	0.1843,	0.1889},
		{0.1606,	0.1765,	0.1766,	0.1835},
		{0.1553,	0.1696,	0.1762,	0.1815},
		{0.1652,	0.1733,	0.1769,	0.1827},
		{0.1875,	0.1841,	0.1823,	0.1823},
		{0.2283,	0.2000,	0.1880,	0.1858},
		{0.2800,	0.2184,	0.1979,	0.1878}
	};

	double r = 0.025;
	double q = 0.0;
	double dt = 0.001, alpha = 0.5;

	//将std::vector转为ublas::vector; 为了统一为boost中的container
	ublas::vector<double> Market_strikes(strikes0.size());
	for (size_t i = 0; i < strikes0.size(); ++i)Market_strikes[i] = strikes0[i];
	ublas::vector<double> Market_maturities(maturities0.size());
	for (size_t i = 0; i < maturities0.size(); ++i)Market_maturities[i] = maturities0[i];
	ublas::matrix<double> Market_impv(impvs0.size(), impvs0[0].size());
	for (size_t i = 0; i < Market_impv.size1(); ++i)
		for (size_t j = 0; j < Market_impv.size2(); ++j)
			Market_impv(i, j) = impvs0[i][j];


	//===数值格式参数===
	double xmin = -2.0, xmax = 2.0, dx = 0.2;
	double zmin = -20.0, zmax = 2.0, dz = 0.1;
	double Tmin = 0.01, Tmax = 1.0, dT = 0.02;
	size_t nx = floor((xmax - xmin) / dx) + 1;
	size_t nz = floor((zmax - zmin) / dz) + 1;
	size_t nt = floor((Tmax - Tmin) / dT) + 1;
	ublas::vector<double> x(nx);
	ublas::vector<double> z(nz);
	ublas::vector<double> Maturity(nt);
	for (size_t i = 0; i < nx; ++i)x[i] = xmin + dx * i;
	for (size_t i = 0; i < nz; ++i)z[i] = zmin + dz * i;
	for (size_t i = 0; i < nt; ++i)Maturity[i] = Tmin + dT * i;

	//initial value of the leverage function
	ublas::matrix<double> L_initial(nx, nt, 1.0);

	double epsilon = 1e-2;
	double step_size = 2e-1;
	int NN = 100;
	double epsilon2 = 1e-2;

	ublas::matrix<double> Leverage_surf;
	ublas::matrix<double> impv_surf;
	ublas::vector<double> x_L;
	calibration_FinitePoint4(x, z, Maturity, L_initial, Market_impv, Market_strikes, Market_maturities, hp, r, q, alpha, epsilon, epsilon2, step_size, NN,
		Leverage_surf, impv_surf, x_L);

	ublas::vector<double> exp_x_L(x_L.size());
	for (auto iter1 = exp_x_L.begin(), iter2 = x_L.begin(); iter1 != exp_x_L.end(); ++iter1, ++iter2)*iter1 = exp(*iter2);

	ublas::matrix<double> Z, X, S, M;
	meshgrid(Maturity, exp_x_L, X, Z);//模型较粗网格
	meshgrid(Market_maturities, Market_strikes, S, M);//市场较疏网格

	ublas::matrix<double> impv_model;
	interp2(X, Z, impv_surf, S, M, impv_model);//将模型surface在市场网格点上进行线性插值
	// double error1;
	// for (int i = 0; i < impv_surf.size1();i++)
	// {
	// 	for (size_t j = 0; j < impv_surf.size2() ;j++)
	// 	{
	// 		if (isnan(impv_surf(i,j)-Market_impv(i,j)))
	// 		{
	// 			error1=error1;
	// 		}
	// 		else{
	// 			error1=error1+pow(impv_surf(i,j)-Market_impv(i,j),2);
	// 		}
	// 	}
		
	// }
	// error1=sqrt(error1);
	// std::cout << "error1="<<error1 << std::endl;
	return 0;
}