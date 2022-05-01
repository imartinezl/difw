#include <iostream>
#include <cmath>
#include <vector>
#include <limits>

// EXPONENTIAL FUNCTION

union di {
	double d;
	uint64_t i;
};

inline unsigned int mask(int x){
	return (1U << x) - 1;
}

inline uint64_t mask64(int x){
	return (1ULL << x) - 1;
}

const int sbit_ = 11;
struct ExpdVar {
	enum {
		sbit = sbit_,
		s = 1UL << sbit,
		adj = (1UL << (sbit + 10)) - (1UL << sbit)
	};
	// A = 1, B = 1, C = 1/2, D = 1/6
	double C1 = 1.0; // A
	double C2 = 0.16666666685227835064; // D
	double C3 = 3.0000000027955394; // C/D
	uint64_t tbl[s];
	double a;
	double ra;
	ExpdVar()
		: a(s / std::log(2.0))
		, ra(1 / a)
	{
		for (int i = 0; i < s; i++) {
			di di;
			di.d = std::pow(2.0, i * (1.0 / s));
			tbl[i] = di.i & mask64(52);
		}
	}
};

static const ExpdVar c;

double fmath_exp(double x){
    if (x <= -708.39641853226408) return 0;
	if (x >= 709.78271289338397) return std::numeric_limits<double>::infinity();

    const uint64_t b = 3ULL << 51;
	di di;
	di.d = x * c.a + b;
	uint64_t iax = c.tbl[di.i & mask(c.sbit)];

	#ifdef __FAST_MATH__
		double tmp1 = di.d * c.ra;
		double tmp2 = b * c.ra;
		double t = tmp1 - tmp2 - x;
	#else
		double t = (di.d - b) * c.ra - x;
	#endif
	uint64_t u = ((di.i + c.adj) >> c.sbit) << 52;
	double y = (c.C3 - t) * (t * t) * c.C2 - t + c.C1;

	di.i = u | iax;
	return y * di.d;
}
double exp(double x){
    return fmath_exp(x);
    // return std::exp(x);
}

// FUNCTIONS
float eps = std::numeric_limits<float>::epsilon();
float inf = std::numeric_limits<float>::infinity();


int sign(const int r){
    return (r > 0) - (r < 0);
}

int signf(const float r){
    return (r > 0) - (r < 0);
}

// TODO: replace 2 for params per cell
bool cmpf(float x, float y)
{
    return std::fabs(x - y) < eps;
}

bool cmpf0(const float& x)
{   
    return std::fabs(x) < eps;
}

float right_boundary(const int& c, const float& xmin, const float& xmax, const int& nc){
    return xmin + (c + 1) * (xmax - xmin) / nc + eps;
}

float left_boundary(const int& c, const float& xmin, const float& xmax, const int& nc){
    return xmin + c * (xmax - xmin) / nc - eps;
}

int get_cell(const float& x, const float& xmin, const float& xmax, const int& nc){
    int c = std::floor((x - xmin) / (xmax - xmin) * nc);
    c = std::max(0, std::min(c, nc-1));
    return c;
}

float get_velocity(const float& x, const float* A, const float& xmin, const float& xmax, const int& nc){
    const int c = get_cell(x, xmin, xmax, nc);
    const float a = A[2*c];
    const float b = A[2*c+1];
    return a*x + b;
}

float derivative_velocity_dx(const float& x, const float* A, const float& xmin, const float& xmax, const int& nc){
    const int c = get_cell(x, xmin, xmax, nc);
    const float a = A[2*c];
    return a;
}

// INTEGRATION CLOSED FORM

float get_psi(const float& x, const float& t, const float& a, const float& b){
    if (cmpf0(a)){
        return x + t*b;
    }
    else{
        const float eta = exp(t*a);
        return eta * x + (b/a) * (eta - 1.0);
        // return exp(t*a) * (x + (b/a)) - (b/a);
    }
}

float get_hit_time(float x, int c, const float& a, const float& b, const float& xmin, const float& xmax, const int& nc, float& xc, int& cc){

    const float v = a * x + b;
    if(cmpf0(v)) return inf;

    cc = c + signf(v);
    if(cc < 0 || cc >= nc) return inf;
    xc = (v > 0) ? right_boundary(c, xmin, xmax, nc) : left_boundary(c, xmin, xmax, nc);

    const float vc = a * xc + b;
    if(cmpf0(vc)) return inf;
    if(signf(v) != signf(vc)) return inf;
    if(xc == xmin || xc == xmax) return inf;

    if(cmpf0(a)){
        return (xc - x)/b;
    }else{
        return std::log(vc / v) / a;
    }
}

float integrate_closed_form(float x, float t, const float* A, const float& xmin, const float& xmax, const int& nc){
    int c = get_cell(x, xmin, xmax, nc);
    int cont = 0;
    const int contmax = std::max(c, nc-1-c);

    float a, b, xc, thit;
    int cc;
    while (true) {
        a = A[2*c];
        b = A[2*c+1];

        thit = get_hit_time(x, c, a, b, xmin, xmax, nc, xc, cc);
        if (thit > t){
            return get_psi(x, t, a, b);
        }

        x = xc;
        c = cc;
        t -= thit;

        cont++;
        if (cont > contmax){
            break;
        }
    }
    return -1;
}

void integrate_closed_form_trace(float* result, float x, float t, const float* A, const float& xmin, const float& xmax, const int& nc){
    int c = get_cell(x, xmin, xmax, nc);
    int cont = 0;
    const int contmax = std::max(c, nc-1-c);

    float a, b, xc, thit;
    int cc;
    while (true) {
        a = A[2*c];
        b = A[2*c+1];

        thit = get_hit_time(x, c, a, b, xmin, xmax, nc, xc, cc);
        if (thit > t){
            result[0] = get_psi(x, t, a, b);
            result[1] = t;
            result[2] = c;
            return;
        }

        x = xc;
        c = cc;
        t -= thit;

        cont++;
        if (cont > contmax){
            break;
        }
    }
    return;
}


// INTEGRATION NUMERIC

float get_psi_numeric(const float& x, const int& c, const float& t, const float* A){
    const float a = A[2*c];
    const float b = A[2*c+1];
    if (cmpf0(a)){
        return x + t*b;
    }
    else{
        return exp(t*a) * (x + (b/a)) - (b/a);
    }
}

float get_phi_numeric(const float& x, const float& t, const int& nSteps2, const float* A, const float& xmin, const float& xmax, const int& nc){
    float yn = x;
    float midpoint;
    const float deltaT = t / nSteps2;
    for(int j = 0; j < nSteps2; j++) {
        midpoint = yn + deltaT / 2 * get_velocity(yn, A, xmin, xmax, nc);
        yn = yn + deltaT * get_velocity(midpoint, A, xmin, xmax, nc);
    }
    return yn;
}

float integrate_numeric(const float& x, const float& t, const float* A, const float& xmin, const float& xmax, const int& nc, const int& nSteps1, const int& nSteps2){
    float xPrev = x;
    const float deltaT = t / nSteps1;
    int c = get_cell(xPrev, xmin, xmax, nc);
    for(int j = 0; j < nSteps1; j++) {
        float xTemp = get_psi_numeric(xPrev, c, deltaT, A);
        int cTemp = get_cell(xTemp, xmin, xmax, nc);
        if (c == cTemp){
            xPrev = xTemp;
        }
        else{
            xPrev = get_phi_numeric(xPrev, deltaT, nSteps2, A, xmin, xmax, nc);
            c = get_cell(xPrev, xmin, xmax, nc);
        }
    }
    return xPrev;
}



// DERIVATIVE

void derivative_psi_theta(float* dpsi_dtheta, const float& x, const int& c, const float& t, const int& d, const float* B, const float* A){
    // int c = get_cell(x, xmin, xmax, nc);
    const double a = A[2*c];
    const double b = A[2*c + 1];


    if (cmpf0(a)){
        for(int k=0; k < d; k++){
            const double ak = B[(2*c)*d + k];
            const double bk = B[(2*c+1)*d + k];
            dpsi_dtheta[k] = t*(x*ak + bk);
        }
    }
    else{
        const double tmp = exp(t*a);
        const double tmp1 = t * tmp * (x + b/a);
        const double tmp2 = (tmp-1)/std::pow(a, 2.0);
        for(int k=0; k < d; k++){
            const double ak = B[(2*c)*d + k];
            const double bk = B[(2*c+1)*d + k];
            dpsi_dtheta[k] = ak * tmp1 + tmp2 * (bk*a - ak*b);
        }
        // return = ak * t * exp(t*a) * (x + b/a) + (exp(t*a)-1)*(bk*a - ak*b)/std::pow(a, 2.0);

    }
}

float derivative_phi_time(const float& x, const int& c, const float& t, const float* A){
    // int c = get_cell(x, xmin, xmax, nc);
    const float a = A[2*c];
    const float b = A[2*c + 1];

    if (cmpf0(a)){
        return b;
    }
    else{
        return exp(t*a)*(a*x + b);
    }
}

void derivative_thit_theta(float* dthit_dtheta_cum, const float& x, const int& c, const float& xc, const int& d, const float* B, const float* A){
    // int c = get_cell(x, xmin, xmax, nc);
    const double a = A[2*c];
    const double b = A[2*c + 1];

    if (cmpf0(a)){
        const double tmp = (x-xc) / std::pow(b, 2.0);
        for(int k=0; k < d; k++){
            const double bk = B[(2*c+1)*d + k];
            dthit_dtheta_cum[k] -= tmp*bk;
        }
    }
    else{
        const double tmp1 = std::log( (a*xc + b) / (a*x + b) )/std::pow(a, 2.0);
        const double tmp2 = (x - xc) / (a * (a*x + b) * (a*xc + b) );
        for(int k=0; k < d; k++){
            const double ak = B[(2*c)*d + k];
            const double bk = B[(2*c+1)*d + k];

            const double d1 = - ak * tmp1;
            const double d2 = ( bk*a - ak*b) * tmp2;
            dthit_dtheta_cum[k] -= d1 + d2; 
        }
        return;
    }
}

void derivative_phi_theta(float* dphi_dtheta, const float& xini, const float& tm, const int& cm, const int& d, const float* B, const float* A, const float& xmin, const float& xmax, const int& nc){ 
    const int cini = get_cell(xini, xmin, xmax, nc);
    float xm = xini;

    float dthit_dtheta_cum[d] = { };
    if (cini != cm){
        float xc;
        const int step = sign(cm - cini);
        for (int c = cini; step*c < cm*step; c += step){
            if (step == 1){
                xc = right_boundary(c, xmin, xmax, nc);
            }else if (step == -1){
                xc = left_boundary(c, xmin, xmax, nc);
            }
            derivative_thit_theta(dthit_dtheta_cum, xm, c, xc, d, B, A);
            xm = xc;
        } 
    }

    const float dpsi_dtime = derivative_phi_time(xm, cm, tm, A);
    float dpsi_dtheta[d] = { };
    derivative_psi_theta(dpsi_dtheta, xm, cm, tm, d, B, A);
    for(int k=0; k < d; k++){
        dphi_dtheta[k] = dpsi_dtheta[k] + dpsi_dtime*dthit_dtheta_cum[k];    
    }
}


// GRADIENT SPACE

float derivative_thit_x(const float& x, const int& c, const float& t, const float* A){
    const float a = A[2*c];
    const float b = A[2*c + 1];

    return 1.0 / (a*x + b);
}

float derivative_psi_x(const float& x, const int& c, const float& t, const float* A){
    const float a = A[2*c];
    // const float b = A[2*c + 1];

    return exp(t*a);
}

float derivative_psi_t(const float& x, const int& c, const float& t, const float* A){
    const float a = A[2*c];
    const float b = A[2*c + 1];

    return exp(t*a)*(a*x + b);
}

float derivative_phi_x(const float& xini, const float& tini, const float& tm, const int& cm, const float* A, const float& xmin, const float& xmax, const int& nc){
    const int cini = get_cell(xini, xmin, xmax, nc);
    float xm = xini;

    float dpsi_dx = 0.0;
    float dthit_dx = 0.0;
    if (cini == cm){
        dpsi_dx = derivative_psi_x(xini, cini, tini, A);
    }else{
        dthit_dx = derivative_thit_x(xini, cini, tini, A);
    }


    if (cini != cm){
        float xc;
        const int step = sign(cm - cini);
        for (int c = cini; step*c < cm*step; c += step){
            if (step == 1){
                xc = right_boundary(c, xmin, xmax, nc);
            }else if (step == -1){
                xc = left_boundary(c, xmin, xmax, nc);
            }
            xm = xc;
        } 
    }

    float dpsi_dtime = derivative_psi_t(xm, cm, tm, A);
    float dphi_dx = dpsi_dx + dpsi_dtime * dthit_dx;
    return dphi_dx;
}

// GRADIENT SPACE DERIVATIVE THETA

float derivative_psi_x_theta(const float& x, const int& c, const float& t, const float* A, const int& k, const int& d, const float* B){
    const float a = A[2*c];
    const float b = A[2*c + 1];

    const double ak = B[(2*c)*d + k];
    const double bk = B[(2*c+1)*d + k];

    return t * exp(t*a) * ak;
}

float derivative_thit_x_theta(const float& x, const int& c, const float& t, const float* A, const int& k, const int& d, const float* B){
    const float a = A[2*c];
    const float b = A[2*c + 1];

    const double ak = B[(2*c)*d + k];
    const double bk = B[(2*c+1)*d + k];

    return - (x*ak + bk)/std::pow(a*x + b, 2.0);
}

float derivative_psi_t_theta(const float& dtm, const float& x, const int& c, const float& t, const float* A, const int& k, const int& d, const float* B){
    const float a = A[2*c];
    const float b = A[2*c + 1];

    const double ak = B[(2*c)*d + k];
    const double bk = B[(2*c+1)*d + k];      
    
    return exp(t*a) * ( a*(a*x+b)*dtm + ak*(t*(a*x+b) + x) + bk);
}

void derivative_phi_x_theta(float* dphi_dx_dtheta, const float& xini, const float& tini, const float& tm, const int& cm, const int& d, const float* B, const float* A, const float& xmin, const float& xmax, const int& nc){
    const int cini = get_cell(xini, xmin, xmax, nc);
    float xm = xini;

    // float dpsi_dx = 0.0;
    float dthit_dx = 0.0;
    float dpsi_dx_dtheta[d] = {};
    float dthit_dx_dtheta[d] = {};
    if (cini == cm){
        // dpsi_dx = derivative_psi_x(xini, cini, tini, A);
        for(int k=0; k < d; k++){
            dpsi_dx_dtheta[k] = derivative_psi_x_theta(xini, cini, tini, A, k, d, B);
            dthit_dx_dtheta[k] = 0.0;
        }
    }else{
        dthit_dx = derivative_thit_x(xini, cini, tini, A);
        for(int k=0; k < d; k++){
            dthit_dx_dtheta[k] = derivative_thit_x_theta(xini, cini, tini, A, k, d, B);
            dpsi_dx_dtheta[k] = 0.0;
        }
    }

    float dthit_dtheta_cum[d] = { };
    if (cini != cm){
        float xc;
        const int step = sign(cm - cini);
        for (int c = cini; step*c < cm*step; c += step){
            if (step == 1){
                xc = right_boundary(c, xmin, xmax, nc);
            }else if (step == -1){
                xc = left_boundary(c, xmin, xmax, nc);
            }
            derivative_thit_theta(dthit_dtheta_cum, xm, c, xc, d, B, A);
            xm = xc;
        } 
    }

    float dpsi_dtime = derivative_psi_t(xm, cm, tm, A);
    float dpsi_dtime_dtheta[d] = {};
    for(int k=0; k < d; k++){
        dpsi_dtime_dtheta[k] = derivative_psi_t_theta(dthit_dtheta_cum[k], xm, cm, tm, A, k, d, B);
        dphi_dx_dtheta[k] = dpsi_dx_dtheta[k] + dpsi_dtime_dtheta[k]*dthit_dx + dpsi_dtime * dthit_dx_dtheta[k];
    }
}


// GRADIENT SPACE DERIVATIVE X

float derivative_thit_x_x(const float& x, const int& c, const float& t, const float* A){
    const float a = A[2*c];
    const float b = A[2*c + 1];

    return - a / std::pow(a*x + b, 2.0);
}

float derivative_psi_t_x(const float& x, const int& c, const float& t, const float* A){
    const float a = A[2*c];
    const float b = A[2*c + 1];

    return a * exp(t*a);
}

float derivative_phi_x_x(const float& xini, const float& tini, const float& tm, const int& cm, const float* A, const float& xmin, const float& xmax, const int& nc){
    const int cini = get_cell(xini, xmin, xmax, nc);
    float xm = xini;

    float dthit_dx = 0.0;
    float dthit_dx_dx = 0.0;
    if (cini != cm){
        dthit_dx = derivative_thit_x(xini, cini, tini, A);
        dthit_dx_dx = derivative_thit_x_x(xini, cini, tini, A);
    }


    if (cini != cm){
        float xc;
        const int step = sign(cm - cini);
        for (int c = cini; step*c < cm*step; c += step){
            if (step == 1){
                xc = right_boundary(c, xmin, xmax, nc);
            }else if (step == -1){
                xc = left_boundary(c, xmin, xmax, nc);
            }
            xm = xc;
        } 
    }

    float dpsi_dtime = derivative_psi_t(xm, cm, tm, A);
    float dpsi_dtime_dx = derivative_psi_t_x(xm, cm, tm, A);
    float dphi_dx = dpsi_dtime_dx * dthit_dx + dpsi_dtime * dthit_dx_dx;
    return dphi_dx;
}