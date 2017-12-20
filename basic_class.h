#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <math.h>

#include <parallel/algorithm>
#include <parallel/numeric>



/*Global constants*/

const double c = 299792458;
const double pi = M_PI;


// Class of Bunch
class bunch {
public:
	double M = 1e6;    /* mass number*/
	double N = 1e6;    /* charge number*/
	double qe = 1.60217662e-19; // elementary charge
	double q = N*qe;
	double me = 1.6726219e-27; // elemetary mass, proton
	double m = M*me;
    double gamma0 = 60; // everage gamma of the bunch. 
    double p0 = sqrt(gamma0*gamma0-1)*me*c;
    
    
	unsigned int Np;


	std::vector<double> x;
	std::vector<double> y;
	std::vector<double> t;
	std::vector<double> px;
	std::vector<double> py;
	std::vector<double> pz;

    std::vector<double> M1; // means of all coordinates;
    std::vector<double> M2; // variances of all coordinates;
    std::vector<double> Emittance; // emittances in three directions;
    

	std::vector<int> index;// holds the sorted indices of the particles, sort is based time coordinates of the particle. 

	// Initialize with given normal distribution parameters
	bunch(unsigned int disType,
		std::vector<double> spc0, // vector that contain centers of distribution in spacial coordinates
		std::vector<double> mmt0, // vector that contain centers of distribution in momentum coordinates
		std::vector<double> spcSig, // vector that contain sigma of distribution in spacial coordinates
		std::vector<double> mmtSig, // vector that contain sigma of distribution in momentum coordinates
		unsigned int Npar,double t0) {
		Np = Npar;
        
		x.resize(Np);
		y.resize(Np);
		t.resize(Np);
		px.resize(Np);
		py.resize(Np);
		pz.resize(Np);
        index.resize(Np);
		M1.resize(6,0);
        M2.resize(6,0);
        Emittance.resize(3,0);
        
		/* So far only the Gaussian distribution is supported */
		std::default_random_engine generator; // Normal distribution generator
		std::normal_distribution<double> dist0(spc0[0], spcSig[0]); // normal distribution in x
		std::normal_distribution<double> dist1(spc0[1], spcSig[1]); // normal distribution in y
		std::normal_distribution<double> dist2(spc0[2], spcSig[2]); // normal distribution in t
		std::normal_distribution<double> dist3(mmt0[0], mmtSig[0]); // normal distribution in px
		std::normal_distribution<double> dist4(mmt0[1], mmtSig[1]); // normal distribution in py
		std::normal_distribution<double> dist5(mmt0[2], mmtSig[2]); // normal distribution in gamma
	    // initialize the particles in one bunch in parallel.
#pragma omp parallel for
		for (int j = 0; j<Np; ++j) {
			x[j] = dist0(generator);
			y[j] = dist1(generator);
			t[j] = dist2(generator);
			px[j] = dist3(generator)*p0;
			py[j] = dist4(generator)*p0;
			pz[j] = (dist5(generator)+1)*p0;
			index[j] = j;
		}
	};

	// Initialize with defualt normal distribution.
	bunch(unsigned int N,double t0) {
		Np = N;
		x.resize(Np);
		y.resize(Np);
		t.resize(Np);
		px.resize(Np);
		py.resize(Np);
		pz.resize(Np);
        index.resize(Np);
        M1.resize(6,0);
        M2.resize(6,0);
        Emittance.resize(3,0);
		/* So far only the Gaussian distribution is supported */
		std::default_random_engine generator; // Normal distribution generator
		std::normal_distribution<double> dist0(0, 1e-3); // normal distribution in x
		std::normal_distribution<double> dist1(0, 1e-3); // normal distribution in y
		std::normal_distribution<double> dist2(0, 1.4e-9); // normal distribution in t
		std::normal_distribution<double> dist3(0, 1e-4); // normal distribution in x'
		std::normal_distribution<double> dist4(0, 1e-4); // normal distribution in y'
		std::normal_distribution<double> dist5(0, 0.00225); // normal distribution in pzs
		// initialize the particles in one bunch in parallel.
#pragma omp parallel for
		for (int j = 0; j<Np; ++j) {
			x[j] = dist0(generator);
			y[j] = dist1(generator);
			t[j] = dist2(generator)+t0;
			px[j] = dist3(generator)*p0;
			py[j] = dist4(generator)*p0;
			pz[j] = (dist5(generator)+1)*p0;//p0;//
			index[j] = j;
		}
		t[0]=t0;
		pz[0]=p0;
		status_update();
	};
// Initialize with defualt normal distribution.
	bunch() {
		Np = 1;
		x.resize(Np);
		y.resize(Np);
		t.resize(Np);
		px.resize(Np);
		py.resize(Np);
		pz.resize(Np);
        index.resize(Np);
        M1.resize(6,0);
        M2.resize(6,0);
        Emittance.resize(3,0);
		/* So far only the Gaussian distribution is supported */
		std::default_random_engine generator; // Normal distribution generator
		std::normal_distribution<double> dist0(0, 1e-3); // normal distribution in x
		std::normal_distribution<double> dist1(0, 1e-3); // normal distribution in y
		std::normal_distribution<double> dist2(0, 1e-15); // normal distribution in t
		std::normal_distribution<double> dist3(0, 1e-3); // normal distribution in px
		std::normal_distribution<double> dist4(0, 1e-3); // normal distribution in py
		std::normal_distribution<double> dist5(0, 1e-3); // normal distribution in gamma
		// initialize the particles in one bunch in parallel.
#pragma omp parallel for
		for (int j = 0; j<Np; ++j) {
			x[j] = dist0(generator);
			y[j] = dist1(generator);
			t[j] = dist2(generator);
			px[j] = dist3(generator)*p0;
			py[j] = dist4(generator)*p0;
			pz[j] = (dist5(generator)+1)*p0;
			index[j] = j;
		}
	};
	
	// Initialize the bunch with PARMELA output file.
	bunch(std::string path,double frq){
	    std::ifstream in;
	    std::string val;
	    std::vector<double> values;
	    in.open(path);
	    getline(in,val);

	    while(in>>val){
	        values.push_back(stod(val));
	    }
	    
	    Np = values.size()/6;
	    x.resize(Np);
		y.resize(Np);
		t.resize(Np);
		px.resize(Np);
		py.resize(Np);
		pz.resize(Np);
        index.resize(Np);
        M1.resize(6,0);
        M2.resize(6,0);
        Emittance.resize(3,0);
        
	    for (int i = 0;i<Np;++i){
	        x[i] = values[6*i];// meter
	        px[i] = values[6*i+1];//rad, will be converted to real momentum later.
			y[i] = values[6*i+2];
			py[i] = values[6*i+3];
			t[i] = values[6*i+4]/180/(2*pi*frq);// raw is degree, convert to time
			pz[i] = values[6*i+5]*1e6*qe;// raw data is kinetic energy in MeV for single micro particle.
			index[i] = i;
	    }
	    std::cout<<"Number of particles per bunch: " <<Np<<std::endl;
	    std::cout<<"Kinetic energy per particle (MeV): "<<std::accumulate(pz.begin(),pz.end(),0.0)/Np/(1e6*qe)<<std::endl;
		for (int i = 0;i<Np;++i){
	        pz[i] = sqrt(pz[i]*pz[i]+2*pz[i]*me*c*c)/c; // convert the kinetic energy to momentum for micro particle.
	        px[i] = px[i]*pz[i]; // convert the transverse momentum to real momentum.
			py[i] = py[i]*pz[i];
	    }

	    p0 = __gnu_parallel::accumulate(pz.begin(),pz.end(),0.0)/Np;
	    std::cout<<p0<<std::endl;
		gamma0 = sqrt(p0*p0/(me*c*c*me)+1);
		std::cout<<gamma0<<std::endl;
	};
	
    // get the sorted index based on the time coordinates of the particles.
    void sort(){
        __gnu_parallel::sort(index.begin(),index.end(),[&](const double& a,const double& b){return (t[a]<t[b]);}
        );
    };
    
    // Get the statistics of the bunch.
    void status_update(){
        M1[0] = __gnu_parallel::accumulate(x.begin(),x.end(),0.0)/Np;
        M1[1] = __gnu_parallel::accumulate(y.begin(),y.end(),0.0)/Np;
        M1[2] = __gnu_parallel::accumulate(t.begin(),t.end(),0.0)/Np;
        M1[3] = __gnu_parallel::accumulate(px.begin(),px.end(),0.0)/Np;
        M1[4] = __gnu_parallel::accumulate(py.begin(),py.end(),0.0)/Np;
        M1[5] = __gnu_parallel::accumulate(pz.begin(),pz.end(),0.0)/Np;
        
        M2[0] = covariance(x,x,M1[0],M1[0]);
        M2[1] = covariance(y,y,M1[1],M1[1]);
        M2[2] = covariance(t,t,M1[2],M1[2]);
        M2[3] = covariance(px,px,M1[3],M1[3])/(M1[5]*M1[5]); //covariance of the angle px/pz
        M2[4] = covariance(py,py,M1[4],M1[4])/(M1[5]*M1[5]); //covariance of the angle py/pz
        M2[5] = covariance(pz,pz,M1[5],M1[5]);     //covariance of the pz.
        
        Emittance[0] = sqrt(M2[0]*M2[3]-covariance(x,px,M1[0],M1[3])*covariance(x,px,M1[0],M1[3])/(M1[5]*M1[5]));
        Emittance[1] = sqrt(M2[1]*M2[4]-covariance(y,py,M1[1],M1[4])*covariance(y,py,M1[1],M1[4])/(M1[5]*M1[5]));        
        Emittance[2] = sqrt(M2[2]*M2[5]-covariance(t,pz,M1[2],M1[5])*covariance(t,pz,M1[2],M1[5]))*(c/qe);   
        /*
        std::cout<<"sig_t^2*sig_Ek^2:"<<M2[2]*M2[5]<<std::endl;
        std::cout<<"sig_t_Ek^2:"<<covariance(t,pz,M1[2],M1[5])*covariance(t,pz,M1[2],M1[5])*(c/qe*c/qe)<<std::endl;
        */
        // Use the first particle as the reference particle.
        p0 = pz[0];
        gamma0 = sqrt(p0*p0/(me*c*c*me)+1);
    }
    
    
    // Dump the moments to file.
    void dump_to_file(std::string path) {
		std::filebuf fb;
		fb.open(path, std::ios::out);
		std::ostream os(&fb);
		os << M1[0] << "," << M1[1] << ","<< M1[2] << ","<< M1[3] << ","<< M1[4] << ","<< M1[5] << "\n"
		<< M2[0] << "," << M2[1] << ","<< M2[2] << ","<< M2[3] << ","<< M2[4] << ","<< M2[5] << "\n"
		<< Emittance[0] << "," << Emittance[1] << ","<< Emittance[2];
		fb.close();
	};
	// dump the whole bunch to a file. file name is specified by 'path'.
	void dump_coords_to_file(std::string path) {
		std::filebuf fb;
		fb.open(path, std::ios::out);
		std::ostream os(&fb);
		for (int i = 0; i<Np; ++i) {
			os << x[i] << "," << y[i] << "," << t[i] << "," << px[i] << "," << py[i] << "," << pz[i] << '\n';
		}
		fb.close();
	};
};

// Class of beam
class beam {
public:
	double delay; // delay between adjustant bunches, [ns]
	double gap; // gap between bunch trains, [ns]
    unsigned int N_bnch; // total number of bunches.
	std::vector<bunch> bnches;

	// Defualt initializer.
	beam(unsigned int N,unsigned int N_bnches_per_train, unsigned int N_trains,double dly, double gp) {
        bnches.resize(N_bnches_per_train*N_trains);
		delay = dly;
		gap = gp;
        for (int i = 0;i<N_trains;++i){
            for (int j = 0;j<N_bnches_per_train;++j){
                bnches[i*N_bnches_per_train+j] = bunch(N,i*gap+(i*N_bnches_per_train+j)*delay);
            }
		}
	};
    // Initialize with PARMELA output file
    beam(std::string path,double frq,unsigned int N_bnches) {
        bnches.resize(N_bnches);
		delay = 1;
		gap = 20;
        for (int i = 0;i<N_bnches;++i){
		    bnches[i] = bunch(path,frq);
		}
	};
	// Initialize with given normal distribution parameters
	beam(
		double dly,
		double gp,
		std::vector<double> spc0, // vector that contain centers of distribution in spacial coordinates
		std::vector<double> mmt0, // vector that contain centers of distribution in momentum coordinates
		std::vector<double> spcSig, // vector that contain sigma of distribution in spacial coordinates
		std::vector<double> mmtSig, // vector that contain sigma of distribution in momentum coordinates
		unsigned int N, unsigned int N_bnches_per_train,unsigned int N_trains) // particles per bunch and bunches per beam.
		{
        bnches.resize(N_bnches_per_train*N_trains);
		delay = dly;
		gap = gp;
		for (int i = 0;i<N_trains;++i){
            for (int j = 0;j<N_bnches_per_train;++j){
		        bnches[i] = bunch(N, spc0, mmt0, spcSig, mmtSig, N,i*gap+(i*N_bnches_per_train+j)*delay);
		    }
		}
	};
};


// Class of cavity
class cavity {
public:
	std::vector<double> frq; // Frequency of the cavity, [MHz]
	std::vector<double> V0xR,V0yR,V0zR; // Real parts of the Voltage of the cavity, [MV]
	std::vector<double> V0xI,V0yI,V0zI; // Imaginary parts of the Voltage of the cavity, [MV]
	
	
    std::vector<std::vector<double>> VxTauR;
	std::vector<std::vector<double>> VyTauR;
	std::vector<std::vector<double>> VzTauR;
	std::vector<std::vector<double>> VxTauI;
	std::vector<std::vector<double>> VyTauI;
	std::vector<std::vector<double>> VzTauI;// vectors to hold the cumulated wake info at each time point (corresponding to each particle), one vector for each mode.

    std::vector<double> kx; 
    std::vector<double> ky;
    std::vector<double> kz;// loss factors for each mode, use this to calculate wake of each mode.
    std::vector<double> tau_invert; // one over tau, save some calculation for each particle.
    std::vector<double> phi; // phase of each mode.
    double t_last = 0; // the time of last particle leaving the cavity from last bunch.
    int N_mod=1;// number of modes to consider
    int bnch_count = 0; // The number of bunches that has already passed the cavity, used to calculate wake field.
    int wake_initialized = 0;
	cavity() {
		frq.resize(N_mod,0.0);
		phi.resize(N_mod,0.0);
		V0xR.resize(N_mod,0.0);
		V0yR.resize(N_mod,0.0);
		V0zR.resize(N_mod,0.0);
		V0xI.resize(N_mod,0.0);
		V0yI.resize(N_mod,0.0);
		V0zI.resize(N_mod,0.0);
		VxTauR.resize(N_mod);
		VyTauR.resize(N_mod);
		VzTauR.resize(N_mod);
		VxTauI.resize(N_mod);
		VyTauI.resize(N_mod);
		VzTauI.resize(N_mod);
		kx.resize(N_mod,0.0);
		ky.resize(N_mod,0.0);
		kz.resize(N_mod,0.0);
		tau_invert.resize(N_mod,0.0);
	};
	
	// Most naive way of calculating the wake, basically sum all the wake from each macro particles for each mode, 
	// Equavlent to convolute the bunch with wake function, slow as hell...
	
	void wake_Naive(beam& bm, bunch& bnch){
	    if (wake_initialized == 0){
	        for (int i = 0;i<N_mod;++i){
	            VxTauR[i].resize(bnch.Np+1,0); // extra one element at the end, use to store the wake in this mode when bunch leaves the cavity.
	            VyTauR[i].resize(bnch.Np+1,0);
	            VzTauR[i].resize(bnch.Np+1,0);
	            VxTauI[i].resize(bnch.Np+1,0); 
	            VyTauI[i].resize(bnch.Np+1,0);
	            VzTauI[i].resize(bnch.Np+1,0);
	        }
	        wake_initialized = 1;
	    }
	    
        int j = 0;
#pragma omp parallel for private(j)
	    for(int i = 0;i<N_mod;++i){ // iterate over number of modes.
	        double Vbx = kx[i]*bnch.q;// always negative(?) imaginary.
	        double Vby = ky[i]*bnch.q;// always negative(?) imaginary.
	        double Vbz = -kz[i]*bnch.q;// always negative real.
	        double dT = bnch.t[bnch.index[0]]-t_last+bm.delay;
       	    //std::cout<<"Delay between bunches:"<<dT<<std::endl;
	        double decay = exp(-dT*tau_invert[i]);// decay of wake from last bunch
	        //std::cout<<"Decay factor of the wake:"<<decay<<std::endl;
	        double dphi = dT*frq[i]*2.0*pi; // phase shift of the wake from last bunch.

	        // rotated wake from last bunch:
	        VxTauR[i][0] = VxTauR[i][bnch.Np]*decay*cos(dphi)-VxTauI[i][bnch.Np]*decay*sin(dphi);
	        VyTauR[i][0] = VyTauR[i][bnch.Np]*decay*cos(dphi)-VyTauI[i][bnch.Np]*decay*sin(dphi);
	        VzTauR[i][0] = VzTauR[i][bnch.Np]*decay*cos(dphi)-VzTauI[i][bnch.Np]*decay*sin(dphi);
	        VxTauI[i][0] = VxTauR[i][bnch.Np]*decay*sin(dphi)+VxTauI[i][bnch.Np]*decay*cos(dphi);
	        VyTauI[i][0] = VyTauR[i][bnch.Np]*decay*sin(dphi)+VyTauI[i][bnch.Np]*decay*cos(dphi);
	        VzTauI[i][0] = VzTauR[i][bnch.Np]*decay*sin(dphi)+VzTauI[i][bnch.Np]*decay*cos(dphi);
	        
	        
	        for (j = 1;j<bnch.Np;++j){ // iterate over every particles. 
	                                   // ith particle sees the wake of all
	                                   // previous particles with a phase shift
	                                   // 2pi*f*(t(i)-t(i-1))
	                                   // ignore the decay of the wake between
	                                   // two macro particles. 
	            int tempID = bnch.index[j];
	            double dt = bnch.t[bnch.index[j]]-bnch.t[bnch.index[j-1]];
	            double cosin = cos(2.0*pi*frq[i]*dt);
	            double sine = sin(2.0*pi*frq[i]*dt);
	            VxTauR[i][j] = VxTauR[i][j-1]*cosin-(VxTauI[i][j-1]+Vbx*bnch.x[tempID])*sine;
	            VyTauR[i][j] = VyTauR[i][j-1]*cosin-(VyTauI[i][j-1]+Vby*bnch.y[tempID])*sine;
	            VzTauR[i][j] = (VzTauR[i][j-1]+Vbz)*cosin-VzTauI[i][j-1]*sine;
	            VxTauI[i][j] = VxTauR[i][j-1]*sine+(VxTauI[i][j-1]+Vbx*bnch.x[tempID])*cosin;
	            VyTauI[i][j] = VyTauR[i][j-1]*sine+(VyTauI[i][j-1]+Vby*bnch.y[tempID])*cosin;
	            VzTauI[i][j] = (VzTauR[i][j-1]+Vbz)*sine+VzTauI[i][j-1]*cosin;
	        }
	        
	        VxTauR[i][bnch.Np] = VxTauR[i][j-1];
	        VyTauR[i][bnch.Np] = VyTauR[i][j-1];
	        VzTauR[i][bnch.Np] = VzTauR[i][j-1]+Vbz;
	        VxTauI[i][bnch.Np] = VxTauI[i][j-1]+Vbx*bnch.x[bnch.index[bnch.Np-1]];
	        VyTauI[i][bnch.Np] = VyTauI[i][j-1]+Vby*bnch.y[bnch.index[bnch.Np-1]];;
	        VzTauI[i][bnch.Np] = VzTauI[i][j-1];
	    }
	    t_last = bnch.t[bnch.index[bnch.Np-1]];

	    
	    /*
	    std::cout<<"Wake after bunch leaves the cavity is: VxR,VxI,VyR,VyI,VzR,VzI= "<< VxTauR[0][bnch.Np]<<","<<VxTauI[0][bnch.Np]<<","<<VyTauR[0][bnch.Np]<<","<<VyTauI[0][bnch.Np]<<","<<VzTauR[0][bnch.Np]<<","<<VzTauI[0][bnch.Np]<<std::endl;
	    */
	};
	
	void update_coord(bunch& bnch) {
	    double qoc = bnch.qe/c;
#pragma omp parallel	    
	    for (int i = 0;i<N_mod;++i){ // iterate over all modes
	        double fi=frq[i];
	        double phiN = phi[i]/180.0*pi;
	        double taui=tau_invert[i];
	        double Vbz = -kz[i]*bnch.q;// always negative real.
#pragma omp for
		    for (int j = 0; j<bnch.Np; ++j) { // iterate over all particles

			    int tempID = bnch.index[j];
			    double dphi = 2.0 * pi*fi*bnch.t[tempID]+phiN;
			    double cosphi = cos(dphi);
			    double sinphi = sin(dphi);
			    //double expi = exp(-(bnch.t[tempID])*taui);
			    // The voltage each particle sees is the real part of the complex voltage in cavity at that time, 
			    // It should be the combination of the cavity voltage (including the previous wake_voltage), and half of the self field(z). 
			    bnch.px[tempID] += qoc*(V0xR[i]*cosphi-V0xI[i]*sinphi+VxTauR[i][j]); // kick in x direction. cannot see self field.
			    bnch.py[tempID] += qoc*(V0yR[i]*cosphi-V0yI[i]*sinphi+VyTauR[i][j]); // kick in y direction.
			    bnch.pz[tempID] += qoc*((V0zR[i]*cosphi-V0zI[i]*sinphi+VzTauR[i][j])+(V0xR[i]*cosphi-V0xI[i]*sinphi)*(2*pi*fi)/c*bnch.x[tempID]+(V0yR[i]*cosphi-V0yI[i]*sinphi)*(2*pi*fi)/c*bnch.y[tempID]+Vbz*0.5); // kick in z direction
		    }
		}
		/*
		std::cout<<"Cavity field after bunch leaves the cavity is: VxR,VxI,VyR,VyI,VzR,VzI= "<<V0xR[0]<<","<<V0xI[0]<<","<<V0yR[0]<<","<<V0yI[0]<<","<<V0zR[0]<<","<<V0zI[0]<<std::endl;
		*/
		// Update the bunch momentum and energy.
		bnch.p0 = std::accumulate(bnch.pz.begin(),bnch.pz.end(),0.0)/bnch.pz.size();
		bnch.gamma0 = sqrt(bnch.p0*bnch.p0/(bnch.me*bnch.me*c*c)+1.0);
	};
	
	void dump_voltage(std::string& path){
	    std::filebuf fb;
		fb.open(path, std::ios::out);
		std::ostream os(&fb);
		int size = VxTauR[0].size()-1;
		for (int i = 0; i<N_mod; ++i) {
			os << V0xR[i]+VxTauR[i][size] << "," << V0xI[i]+VxTauI[i][size] << "," << V0yR[i]+VyTauR[i][size] << "," << V0yI[i]+VyTauI[i][size] << "," << V0zR[i]+VzTauR[i][size] << "," << V0zI[i]+VzTauI[i][size] << '\n';
		}
		fb.close();
	}
};

// Class of ring
class ring {
public:
	double alpx, alpy, betax, betay, gammax, gammay; // ring lattice parameters.
	double phix, phiy;
	double R;   // The radius of the ring.
	double f0;
	double GAMTSQ; // Transition gamma.
	double cosphix;
	double sinphix;
	double cosphiy;
	double sinphiy;
	double *TM;//Transfer matrix.
	ring() {
		R = 610.1754; // RHIC ring.
		GAMTSQ = 691.69;//691.69;
		f0 = 0;
		alpx = 0;
		alpy = 0;
		betax = 1;
		betay = 1;
		gammax = 0;
		gammay = 0;
		phix = 2;
		phiy = 2;
		cosphix = cos(phix);
		sinphix = sin(phix);
		cosphiy = cos(phiy);
		sinphiy = sin(phiy);
		TM = new double[6 * 6];
		for (unsigned int i = 0; i < 36; ++i) {
			TM[i] = 0;
		}
		TM[0] = betax*(cosphix + alpx*sinphix);
		TM[1] = betax*sinphix;
		TM[6] = -(1 + alpx*alpx) / betax*sinphix;
		TM[7] = cosphix;
		TM[14] = betay*(cosphiy + alpy*sinphiy);
		TM[15] = betay*sinphiy;
		TM[20] = -(1 + alpy*alpy) / betay*sinphiy;
		TM[21] = cosphiy;
	};
	void updt_coord_blas(double* bnch) {

	};
	void update_f0(bunch& bnch){
	    double GM0SQ_inv = 1.0/(bnch.gamma0*bnch.gamma0);
	    f0 = (c*sqrt(1.0-GM0SQ_inv))/(2.0*pi*R);
	};
	void update_coord(bunch& bnch) {
	    double alpha0 = 1.0/GAMTSQ;
	    double GM0SQ_inv = 1.0/(bnch.gamma0*bnch.gamma0);
	    double p0_inv = 1.0/bnch.p0;

	    double f0_inv = (2.0*pi*R)/(c*sqrt(1.0-GM0SQ_inv)); // Revolution time.
        /*
        std::cout<< "bnch.pz[i] "<<bnch.pz[0]<<std::endl;
        std::cout<< "bnch.px[i] "<<bnch.px[0]<<std::endl;     
        */
#pragma omp parallel for        
        // zeroth order transportation
		for (int i = 0; i<bnch.Np; ++i) {
		    double delta = (bnch.pz[i]*p0_inv-1.0);
		    double x = bnch.x[i];
		    double y = bnch.y[i];
		    double px = bnch.px[i];
		    double py = bnch.py[i];
		    double pz_inv = 1/bnch.pz[i];
		    double ita = alpha0-GM0SQ_inv;//+0.01*delta+0.001*delta*delta;
			bnch.x[i] = x * (cosphix + alpx*sinphix) + betax*sinphix*px*pz_inv;//bnch.x[i];//
			bnch.y[i] = y * (cosphiy + alpy*sinphiy) + betay*sinphiy*py*pz_inv;//bnch.y[i];//
			bnch.t[i] += f0_inv/(-1.0+1.0/((ita)*delta));//f0_inv;//f0_inv/(1.0-1.0/((ita)*delta));
			bnch.px[i] = -x * (1.0 + alpx*alpx) / betax*sinphix *bnch.pz[i]+ cosphix*px;//bnch.px[i];//
			bnch.py[i] = -y * (1.0 + alpy*alpy) / betay*sinphiy *bnch.pz[i]+ cosphiy*py;//bnch.py[i];//
			bnch.pz[i] += 0;
		}

	};
};

// Class of drift space
class drift_space{
public:
    double L;// length of the drift space.
    drift_space(double dft_L = 1.73){
        L = dft_L;// Default length is 1 m.
    }
    void update_coord(bunch& bnch){
        double p0_invert = 1/bnch.p0;

#pragma omp paprallel for 
        for (int i = 0;i<bnch.Np;++i){
            bnch.x[i] += L*bnch.px[i]*p0_invert;
            bnch.y[i] += L*bnch.py[i]*p0_invert;
        }
    }
};
// Class of lattice
class lattice {
public:
};

// Class of input parameters
class inputPara{
public:
    double* frqs; // frequencies of each mode in a cavity.
    double* V0s; // Existing voltage of each mode in a cavity at the beginning.
    double* RoQs; // R over Qs of each mode in a cavity.
    double* Qs; // Loaded Qs of each mode in a cavity.
    double* ks; // loss factors of each mode in a cabity.
    double R=610.1754; // radius of the ring.
    double gammaT; // Transision energy of the ring.
};

// Class of output buffer
class outputs{
public:
    unsigned int recordLength;
    std::vector<double> gamma;
    std::vector<double> sig_t;
    std::vector<double> sig_Ek;
    std::vector<double> sig_x;
    std::vector<double> sig_xprime;
    std::vector<double> sig_y;
    std::vector<double> sig_yprime;
    std::vector<double> emittance;
    std::vector<unsigned int> turn_stamp;
    outputs(unsigned int N_records){
        recordLength = N_records;
        gamma.resize(N_records,0);
        sig_t.resize(N_records,0);
        sig_Ek.resize(N_records,0);
        sig_x.resize(N_records,0);
        sig_xprime.resize(N_records,0);
        sig_y.resize(N_records,0);
        sig_yprime.resize(N_records,0);
        emittance.resize(N_records*3,0);
        turn_stamp.resize(N_records,0);   
    };
    void update(bunch& bnch,unsigned int n_of_records,unsigned int turns_per_record){
            gamma[n_of_records] = bnch.gamma0;
	        sig_t[n_of_records] = sqrt(bnch.M2[2]);
	        sig_Ek[n_of_records] = sqrt(bnch.M2[5]);
	        sig_x[n_of_records] = sqrt(bnch.M2[0]);
	        sig_xprime[n_of_records] = sqrt(bnch.M2[3]);
	        sig_y[n_of_records] = sqrt(bnch.M2[1]);
	        sig_yprime[n_of_records] = sqrt(bnch.M2[4]);
	        emittance[n_of_records*3] = bnch.Emittance[0];
	        emittance[n_of_records*3+1] = bnch.Emittance[1];
	        emittance[n_of_records*3+2] = bnch.Emittance[2];
	        turn_stamp[n_of_records] = n_of_records*turns_per_record;
    }
    
    void dump_to_file(std::string& path){
	    std::filebuf fb;
		fb.open(path, std::ios::out);
		std::ostream os(&fb);
		os<<std::left<<std::setw(20)<<"Turn stamp";
		os<<std::left<<std::setw(20)<<"Gamma";
		os<<std::left<<std::setw(20)<<"sig_t (s)";
		os<<std::left<<std::setw(20)<<"sig_Ek (eV)";
		os<<std::left<<std::setw(20)<<"sig_x (m)";
		os<<std::left<<std::setw(20)<<"sig_xprime (rad)";
        os<<std::left<<std::setw(20)<<"sig_y (m)";
		os<<std::left<<std::setw(20)<<"sig_yprime (rad)";
		os<<std::left<<std::setw(20)<<"Emittance_x (m-rad)";
		os<<std::left<<std::setw(20)<<"Emittance_y (m-rad)";		
		os<<std::left<<std::setw(20)<<"Emittance_z (eV.s)"<<std::endl;
		for (int i = 0; i<recordLength; ++i) {
		    os<<std::left<<std::setw(20)<<turn_stamp[i];
		    os<<std::left<<std::setw(20)<<gamma[i];
		    os<<std::left<<std::setw(20)<<sig_t[i];
		    os<<std::left<<std::setw(20)<<sig_Ek[i];
		    os<<std::left<<std::setw(20)<<sig_x[i];
		    os<<std::left<<std::setw(20)<<sig_xprime[i];
		    os<<std::left<<std::setw(20)<<sig_y[i];
		    os<<std::left<<std::setw(20)<<sig_yprime[i];
		    os<<std::left<<std::setw(20)<<emittance[i*3];
		    os<<std::left<<std::setw(20)<<emittance[i*3+1];
		    os<<std::left<<std::setw(20)<<emittance[i*3+2]<<std::endl;
		}
		fb.close();
	}
};

