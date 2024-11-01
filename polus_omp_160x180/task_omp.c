#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include <omp.h>

int M,N,nDim;
const double A1 = -3, B1 = 3;    // x范围
const double A2 = 0, B2 = 3;    // y范围
double h1,h2,h;


int isInD(double x, double y) {
    // Trapezoid vertices: A(-3,0), B(3,0), C(2,3), D(-2,3)
    if (y < 0 || y > 3) return 0;
    if (y == 0 && x >= -3 && x <= 3) return 1;
    if (y == 3 && x >= -2 && x <= 2) return 1;
    double x_left = -3 + y/3;
    double x_right = 3 - y/3;
    return (x >= x_left && x <= x_right);
}

double func_k(double x, double y){
    if (isInD(x, y)) return 1.0;
    return 1.0 / (h * h); // ε = h²
}



double func_F(double x, double y){
    if (isInD(x, y)) return 1.0;
    return 0.0;
}

int tot_iteration = 0;
double initial_value = 1;
double eps = 0.000001;

int** MatrA_index;
double** MatrA_val;
double** Matr_map;
double* VecB;
double* VecRes;

int getindex_x(int x){
    return x % (M-1);
}
int getindex_y(int x){
    return x / (M-1);
}
double get_x(int x){
    return A1 + h1 * (getindex_x(x) + 1);
}
double get_y(int x){
    return B1 + h2 * (getindex_y(x) + 1);
}

struct CSR{
    int* IA;
    int* JA;
    double* A;
}MatrCSR;

void FreeMatrCSR(struct CSR Matr){
	free(Matr.IA);
	free(Matr.JA);
	free(Matr.A);
}

void FreeMatr_int(int** Matr, int nDim, int mDim){
	for (int i = 0; i < nDim; i++){
		free(Matr[i]);
	}
	free(Matr);
}
void FreeMatr_double(double** Matr, int nDim, int mDim){
	for (int i = 0; i < nDim; i++){
		free(Matr[i]);
	}
	free(Matr);
}

void PrintMatr_file(double** Matr, int m, int n, char* filename){
  FILE* fout;
  if ((fout = fopen(filename, "w")) == NULL) {   //_CRT_SECURE_NO_WARNINGS
    printf("Woring open file %s!\n", filename);
    exit(-1);
  }

  for (int i = 0; i < m; i++){
    for (int j = 0 ; j < n; j++){
      fprintf(fout,"%lf",Matr[i][j]);
      if (j!=n-1) fprintf(fout," ");
    }
    fprintf(fout,"\n");
  } 

  fclose(fout);
}

void Init_Matr_Vec(int nDim){
	MatrA_index = malloc(nDim*sizeof(int*));
	MatrA_val = malloc(nDim*sizeof(double*));

	for (int i = 0; i < nDim; i++){
		MatrA_index[i] = malloc(5*sizeof(int));
		MatrA_val[i] = malloc(5*sizeof(double));
		for (int j = 0; j < 5; j++){
			MatrA_index[i][j] = -1;
			MatrA_val[i][j] = 0;
		}
	} 



	VecB = malloc(nDim*sizeof(double));
	for (int i = 0; i < nDim; i++) VecB[i] = 0;
}

// Helper function to calculate a_ij
double calculate_aij(int i, int j) {
    double y_j_minus_half = A2 + h2 * (j - 0.5);
    double y_j_plus_half = A2 + h2 * (j + 0.5);
    double x_i_minus_half = A1 + h1 * (i - 0.5);
    
    // Numerical integration using midpoint rule
    int num_points = 100;
    double dy = (y_j_plus_half - y_j_minus_half) / num_points;
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (int k = 0; k < num_points; k++) {
        double y = y_j_minus_half + (k + 0.5) * dy;
        sum += func_k(x_i_minus_half, y) * dy;
    } 

    
    return sum / h2;
}

// Helper function to calculate b_ij
double calculate_bij(int i, int j) {
    double x_i_minus_half = A1 + h1 * (i - 0.5);
    double x_i_plus_half = A1 + h1 * (i + 0.5);
    double y_j_minus_half = A2 + h2 * (j - 0.5);
    
    // Numerical integration using midpoint rule
    int num_points = 100;
    double dx = (x_i_plus_half - x_i_minus_half) / num_points;
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)	
    for (int k = 0; k < num_points; k++) {
        double x = x_i_minus_half + (k + 0.5) * dx;
        sum += func_k(x, y_j_minus_half) * dx;
    }
    
    return sum / h1;
}

// Helper function to calculate F_ij
double calculate_Fij(int i, int j) {
    double x_i_minus_half = A1 + h1 * (i - 0.5);
    double x_i_plus_half = A1 + h1 * (i + 0.5);
    double y_j_minus_half = A2 + h2 * (j - 0.5);
    double y_j_plus_half = A2 + h2 * (j + 0.5);
    
    // Numerical integration using midpoint rule
    int num_points = 20;
    double dx = (x_i_plus_half - x_i_minus_half) / num_points;
    double dy = (y_j_plus_half - y_j_minus_half) / num_points;
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum) collapse(2)
    for (int k = 0; k < num_points; k++) {
        for (int l = 0; l < num_points; l++) {
            double x = x_i_minus_half + (k + 0.5) * dx;
            double y = y_j_minus_half + (l + 0.5) * dy;
            sum += func_F(x, y) * dx * dy;
        }
    }
    
    return sum / (h1 * h2);
}

void Filling_MatrA_VecB() {
	

    for (int j = 0; j < N-1; j++) {
        for (int i = 0; i < M-1; i++) {
            int current = i + j * (M-1);
            int idx = 0;
            
            double aij = calculate_aij(i+1, j+1);
            double ai1j = calculate_aij(i+2, j+1);
            double bij = calculate_bij(i+1, j+1);
            double bij1 = calculate_bij(i+1, j+2);
            
            // Diagonal element
            double diag = (ai1j + aij)/(h1*h1) + (bij1 + bij)/(h2*h2);
            MatrA_val[current][idx] = diag;
            MatrA_index[current][idx] = current;
            idx++;
            
            // Left neighbor
            if (i > 0) {
                MatrA_val[current][idx] = -aij/(h1*h1);
                MatrA_index[current][idx] = current-1;
                idx++;
            }
            
            // Right neighbor
            if (i < M-2) {
                MatrA_val[current][idx] = -ai1j/(h1*h1);
                MatrA_index[current][idx] = current+1;
                idx++;
            }
            
            // Bottom neighbor
            if (j > 0) {
                MatrA_val[current][idx] = -bij/(h2*h2);
                MatrA_index[current][idx] = current-(M-1);
                idx++;
            }
            
            // Top neighbor
            if (j < N-2) {
                MatrA_val[current][idx] = -bij1/(h2*h2);
                MatrA_index[current][idx] = current+(M-1);
                idx++;
            }
            
            // Right hand side vector
            VecB[current] = calculate_Fij(i+1, j+1);
        }
    }
}

struct CSR Change_Matr_to_CSR(int** MatrA_index,double** MatrA_val, int nDim){
	struct CSR MatrCSR;
	MatrCSR.IA = malloc((nDim+1)*sizeof(int));
	for (int i = 0; i < nDim+1; i++){
			MatrCSR.IA[i] = 0;
	}

	int nsize = 0;
	for (int i = 0; i < nDim; i++){
		for (int j = 0; j < 5; j++)
			if (MatrA_index[i][j]!= -1) nsize++;
		MatrCSR.IA[i+1] = nsize;
	}

	MatrCSR.JA = malloc((nsize)*sizeof(int));
	MatrCSR.A = malloc((nsize)*sizeof(double));

	for (int i = 0; i < nDim; i++){
		int k = MatrCSR.IA[i];
		for (int j = 0; j < 5; j++){
			if (MatrA_index[i][j] == -1) continue;
			MatrCSR.JA[k] = MatrA_index[i][j];
			MatrCSR.A[k] = MatrA_val[i][j];
			k++; 
		}
	}
	return MatrCSR;
}

//OpenMP functuion
void SpMV(struct CSR MatrCSR, double* x, double* ans) {
    #pragma omp parallel for
    for (int i = 0; i < nDim; i++) {
        double sum = 0.0;
        const int jb = MatrCSR.IA[i];
        const int je = MatrCSR.IA[i + 1];
        for (int j = jb; j < je; j++) {
            sum += MatrCSR.A[j] * x[MatrCSR.JA[j]];
        }
        ans[i] = sum;
    }
}

void Axpy(double* x, double* y, double a, int flag, int nDim) {
    
    if (flag == 1) {
        #pragma omp parallel for //schedule(dynamic,100)
        for (int i = 0; i < nDim; i++) x[i] = x[i] + a * y[i];
    } else if (flag == 2) {
        #pragma omp parallel for schedule(dynamic,100)
        for (int i = 0; i < nDim; i++) y[i] = x[i] + a * y[i];
    } else {
        printf("error: Axpy MPI flag\n");
        exit(-1);
    }
}

void Vcopy(double* x, double* y, int nDim) {
    #pragma omp parallel for
    for (int i = 0; i < nDim; i++) y[i] = x[i];
}

void Vdiff(double* x, double* y, double* z, int nDim) {
    #pragma omp parallel for
    for (int i = 0; i < nDim; i++) z[i] = x[i] - y[i];
}

double Dot(double * VecA, double* VecB, int nDim) {
    double res = 0.0;
    #pragma omp parallel for reduction(+:res)
    for (int i = 0; i < nDim; i++) {
        res += VecA[i] * VecB[i];
    }
    return res;
}

double Norm_Vec(double* Vec, int nDim) {
    return sqrt(Dot(Vec, Vec, nDim));
}

double* Solve_SOLE_Parallel(struct CSR MatrCSR, double* VecB, double init_val, int nDim) {
    double* VecRes = malloc(nDim * sizeof(double));
    double* VecR = malloc(nDim * sizeof(double));
    double* VecAr = malloc(nDim * sizeof(double));
    double* Vec0 = malloc(nDim * sizeof(double));
    double* tmp = malloc(nDim * sizeof(double));

    #pragma omp parallel for
    for (int i = 0; i < nDim; i++) {
        VecRes[i] = init_val;
        VecR[i] = init_val - VecB[i];
        VecAr[i] = 0;
        Vec0[i] = 0;
    }

    double tau = 0.0;
    tot_iteration = 0;
    do {
        // Ar:
        SpMV(MatrCSR, VecR, VecAr);

        // tau:
        tau = (Dot(VecAr, VecR, nDim) / Dot(VecAr, VecAr, nDim));

        // Vec0:
        Vcopy(VecRes, Vec0, nDim);

        // wij:
        Axpy(VecRes, VecR, -tau, 1, nDim);

        // r:
        SpMV(MatrCSR, VecRes, VecR);
        Axpy(VecR, VecB, -1, 1, nDim);

        Vdiff(VecRes, Vec0, tmp, nDim);

        tot_iteration++;

    } while (Norm_Vec(tmp, nDim) > eps);

    free(VecR);
    free(VecAr);
    free(Vec0);
    free(tmp);
    return VecRes;
}

int Check_SOLE(struct CSR MatrCSR, double* VecRes, double* VecB) {
    #pragma omp parallel for
    for (int i = 0; i < nDim; i++) {
        double sum = 0.0;
        const int jb = MatrCSR.IA[i];
        const int je = MatrCSR.IA[i + 1];
        for (int j = jb; j < je; j++) {
            sum += MatrCSR.A[j] * VecRes[MatrCSR.JA[j]];
        }
        if (fabs(VecB[i] - sum) > 0.1) {
            printf("%d: %lf %lf eps:%lf\n", i, sum, VecB[i], fabs(VecB[i] - sum));
        }
    }
    return -1;
}

double** Get_map(double* VecRes,int M, int N){
	Matr_map = malloc((M+1)*sizeof(double*));
	for (int i = 0; i < M + 1; i++){
		Matr_map[i] = malloc((N+1)*sizeof(double));
	}
	for (int i = 0; i < M + 1; i++){
		double x = A1 + h1 * i;
		Matr_map[i][0] = 0;	
		Matr_map[i][N] = 0;	
	}
	for (int j = 1; j < N; j++){
		double y = B1 + h2 * j;
		Matr_map[0][j] = 0;	
		Matr_map[M][j] = 0;	
	}
	for (int i = 0;i < nDim; i++){
		int index_x = getindex_x(i) + 1;
		int index_y = getindex_y(i) + 1;
		Matr_map[index_x][index_y] = VecRes[i];
	}
	return Matr_map;
}

int main(int argc,char** argv) {
	assert(argc==8);

	M = atoi(argv[1]);
	N = atoi(argv[2]);
	initial_value = atof(argv[3]);
	eps = atof(argv[4]);
	char* fout = (argv[5]);
	char* ftime = (argv[6]);

//omp
    int num_threads = atoi(argv[7]);
    omp_set_num_threads(num_threads);


	double Tbegin, Tend;

//	Tbegin = clock();
    Tbegin = omp_get_wtime();

 	h1 = (B1 - A1) / M;
	h2 = (B2 - A2) / N; 
	h = fmax(h1,h2);

	nDim = (M-1)*(N-1);
	
	Init_Matr_Vec(nDim);
	
	Filling_MatrA_VecB();

	MatrCSR = Change_Matr_to_CSR(MatrA_index,MatrA_val,nDim);

//omp
	VecRes = Solve_SOLE_Parallel(MatrCSR,VecB,initial_value,nDim);

	/*if (Check_SOLE(MatrCSR,VecRes,VecB) != -1){
		printf("Check: No\n");
	} else{
		printf("Check: Yes\n");
	}*/

	Matr_map = Get_map(VecRes,M,N);

	//Tend = clock();
    Tend = omp_get_wtime();

	printf("tot iteration: %d\n",tot_iteration);
	printf("The run time is: %fs\n", (Tend - Tbegin));

	FILE* file;
	if ((file = fopen(ftime, "w")) == NULL) {	 
		printf("Woring open file %s!\n", ftime);
		exit(-1);
	}
	fprintf(file,"tot iteration: %d\n",tot_iteration);
	fprintf(file,"The run time is: %fs\n", (Tend - Tbegin));


	PrintMatr_file(Matr_map, M+1, N+1, fout);

	
	FreeMatr_int(MatrA_index, M, 5);
	FreeMatr_double(MatrA_val, M, 5);
	free(VecRes);
	free(VecB);
	FreeMatrCSR(MatrCSR);
	FreeMatr_double(Matr_map, M+1, N+1);

	return 0;
}
