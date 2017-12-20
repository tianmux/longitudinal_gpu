#include <vector>
#include <iostream>
#include <ctime>
#include <fstream>
#include <omp.h>
#include "ult.h"
#include "basic_class.h"
#include "graph.h"
#include <chrono>

/*
double max(double a,double b){
    return a>=b?a:b;
}
*/
int main() {
	unsigned int N = 1e3;   // particles per bunch.
	unsigned int N_bnches_p_train = 1; // bunches per train.
	unsigned int N_trns = 1e0; // number of trains
	unsigned int N_turns = 1e0;
	unsigned int N_steps_btwn_records = 1; // number of turns between records.
	double dly = 1/704e6;
	double gp = 0/704e6;
	std::vector<double> kick{ 1,2,4 };
	std::cout.precision(17);
    plot p;
	// Try Initialize a bunch
	std::cout << "Start initializing the bunch." << std::endl;
	std::cout << "Number of particles:" << N << std::endl;
	double start_time = omp_get_wtime();
	std::vector<double> spc0{ 0,0,0 };
	std::vector<double> spcSig{ 2e-3,2e-3,2e-3 };// beam size;
	std::vector<double> mmt0{ 0,0,0 };
	std::vector<double> mmtSig{ 2e-3,2e-3,2e-3 };// beam angle spread
	unsigned int dist_type = 1;
	bunch b1 = bunch(dist_type, spc0, mmt0, spcSig, mmtSig, N,0.0);
	double time = omp_get_wtime() - start_time;
	std::cout << "Time spend on bunch initialzation (openMP):" << time * 1000 << " ms" << std::endl;
    /*
	// Try initialize a beam with PARMELA output
	start_time = omp_get_wtime();
	std::string parmelaData = "Dist_in_defl_cavity.txt";
	beam bm1 = beam(parmelaData,704e6,N_bnches);
	time = omp_get_wtime() - start_time;
	std::cout << "Time spend on initialize a beam with PARMELA output file:" << time * 1000 << " ms" << std::endl;
	*/
	
	// Try initialize a beam

	start_time = omp_get_wtime();

	beam bm1 = beam(N,N_bnches_p_train, N_trns,dly,gp);

	time = omp_get_wtime() - start_time;
	std::cout << "Time spend on initialize a beam: " << time * 1000 << " ms" << std::endl;
	/*
	// Try initialize a bunch from a PARMELA ouput file
	start_time = omp_get_wtime();
	std::string parmelaData = "Dist00.txt";
	bunch bnch2 = bunch(parmelaData,704e6);
	time = omp_get_wtime() - start_time;
	std::cout << "Time spend on initialize a bunch from PARMELA file:" << time * 1000 << " ms" << std::endl;
	*/
	
	/*
	start_time = omp_get_wtime();
	std::string path = "data1.txt";
	bm1.bnch.dump_to_file(path);
	time = omp_get_wtime() - start_time;
	std::cout << "Time spend on dumping the bunch to file (serial):" << time * 1000 << " ms" << std::endl;
	*/

	// Try initialize cavity, ring and drift space.
	std::cout<<"Initializing the cavities and ring..."<<std::endl;
	cavity cvt = cavity();
	cavity cvt2 = cavity();
	cvt.V0zR[0] = 0;
	cvt.V0zI[0] = 1e6;//-1e5;
	cvt.phi[0] = 10;
	cvt.kz[0] = 0;//1.10584061406e11*2;// 1/4*(w*R/Q)
	cvt.tau_invert[0] =0;//1/4.521447e-5; // 2 Q/w
	
	cvt2.V0zR[0] = 0;
	cvt2.V0zI[0] = 0;
	cvt2.phi[0] = 0;
	
	ring rng = ring();
	drift_space dft = drift_space(1.73);
    std::cout<<"Initialize cavity and ring successfully."<<std::endl;
    
    // Try initialize the output buffer;
    outputs otpt = outputs(N_turns/N_steps_btwn_records);
    
	// Try to iterate all particles in a bunch to update and apply the kicks
	std::string path = "data";
    /*
	bm1.bnches[0].dump_to_file(path+std::to_string(0)+std::to_string(0)+"before");
	bm1.bnches[0].dump_coords_to_file(path+std::to_string(0)+std::to_string(0)+"before_coords");
	*/
	std::vector<double> temp_t;
	std::vector<double> temp_pz;
	std::vector<double> temp_vc;
	start_time = omp_get_wtime();
    auto start = std::chrono::high_resolution_clock::now();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
	//std::cout<<bm1.bnches[0].p0<<std::endl;
	start_time = omp_get_wtime();
	for (unsigned int i = 0; i<N_turns; ++i) {

	    rng.update_f0(bm1.bnches[0]);
	    cvt.frq[0]=rng.f0*120;
	    cvt2.frq[0]=rng.f0*240;
	    bm1.delay = 1.0/rng.f0;
	//    std::cout<<"cvt frq = "<< cvt.frq[0] <<std::endl;
    //    std::cout<<"Beam energy: "<<(bm1.bnches[0].gamma0-60.0)*bm1.bnches[0].me*c*c/bm1.bnches[0].qe<<std::endl;
    //    cvt.V0zI[0] -= 1e6/N_turns;
    //    cvt2.V0zI[0] -=3e6/N_turns;
	    for (unsigned int j = 0;j< N_bnches_p_train*N_trns;++j){
	    
	        start = std::chrono::high_resolution_clock::now();
	        bm1.bnches[j].sort();
	        finish = std::chrono::high_resolution_clock::now();
	        elapsed = finish - start;
	        std::cout<<"Time spent on sorting: "<< elapsed.count()*1000<<" ms."<<std::endl;
	        
    //	    std::cout<<"Calculating wake..."<<std::endl;
            start = std::chrono::high_resolution_clock::now();
            cvt.wake_Naive(bm1,bm1.bnches[j]);
            finish = std::chrono::high_resolution_clock::now();
            elapsed = finish - start;
	       std::cout<<"Time spent on waking: "<< elapsed.count()*1000<<" ms."<<std::endl;
		   
		    start = std::chrono::high_resolution_clock::now();
            cvt.update_coord(bm1.bnches[j]);
            finish = std::chrono::high_resolution_clock::now();
            elapsed = finish - start;
	        std::cout<<"Time spent on kicking: "<< elapsed.count()*1000<<" ms."<<std::endl;
		    
	//	    cvt2.wake_Naive(bm1,bm1.bnches[j]);
	//	    cvt2.update_coord(bm1.bnches[j]);
	
	        start = std::chrono::high_resolution_clock::now();
		    rng.update_coord(bm1.bnches[j]);
            finish = std::chrono::high_resolution_clock::now();
            elapsed = finish - start;
	        std::cout<<"Time spent on going around the ring: "<< elapsed.count()*1000<<" ms."<<std::endl;
		    

	//	    dft.update_coord(bm1.bnches[j]);
	//	    temp_t.push_back(bm1.bnches[0].t[0]*2*pi*cvt.frq[0]);
	        start = std::chrono::high_resolution_clock::now();
	        bm1.bnches[j].status_update();
            finish = std::chrono::high_resolution_clock::now();
            elapsed = finish - start;
	        std::cout<<"Time spent on update bunch moments: "<< elapsed.count()*1000<<" ms."<<std::endl;
		    

	        start = std::chrono::high_resolution_clock::now();
            temp_t.push_back(bm1.bnches[0].t[0]);
            temp_pz.push_back((bm1.bnches[0].pz[0]/bm1.bnches[0].M1[5]-1));
            temp_vc.push_back(sqrt(cvt.VzTauR[0][bm1.bnches[0].x.size()]*cvt.VzTauR[0][bm1.bnches[0].x.size()]+cvt.VzTauI[0][bm1.bnches[0].x.size()]*cvt.VzTauI[0][bm1.bnches[0].x.size()]));
            finish = std::chrono::high_resolution_clock::now();
            elapsed = finish - start;
	        std::cout<<"Time spent on recording the results in buffer: "<< elapsed.count()*1000<<" ms."<<std::endl;
	//	    std::cout<<std::endl;
	        
            
            if (i>10 && (i*N_bnches_p_train*N_trns+j)%(N_bnches_p_train*N_trns*N_turns/10)==0){
	            std::cout<<int(float(i*N_bnches_p_train*N_trns+j)/float(N_bnches_p_train*N_trns*N_turns)*100)<<"%..."<<std::endl;
	        }
	    }
	    
	    if(i%N_steps_btwn_records==0){
	        otpt.update(bm1.bnches[0], i/N_steps_btwn_records,N_steps_btwn_records);
	        double xrange = max(abs(max_para(bm1.bnches[0].t)-bm1.bnches[0].M1[2]),abs(min_para(bm1.bnches[0].t)-bm1.bnches[0].M1[2]));
	        double yrange = max(abs(max_para(bm1.bnches[0].pz)-bm1.bnches[0].M1[5]),abs(min_para(bm1.bnches[0].pz)-bm1.bnches[0].M1[5]));
	        //p.plot_data(bm1.bnches[0].t,bm1.bnches[0].pz, bm1.bnches[0].M1[2]+xrange, bm1.bnches[0].M1[5]+yrange, bm1.bnches[0].M1[2]-xrange, bm1.bnches[0].M1[5]-yrange,i*bm1.delay);
	        //p.plot_data(bm1.bnches[0].t,bm1.bnches[0].pz, bm1.bnches[0].M1[2]+1/cvt2.frq[0], bm1.bnches[0].M1[5]*(1+0.01), bm1.bnches[0].M1[2]-1/cvt2.frq[0], bm1.bnches[0].M1[5]*(1-0.01),i*bm1.delay);
	    }
	}
	time = omp_get_wtime() - start_time;
	std::cout << "Total time spend on tracking: "<< time*1000<<" ms."<<std::endl;
	std::cout << "Time spend on finish kick on one bunch:" << time * 1000 / N_turns/N_bnches_p_train/N_trns << " ms." << std::endl;
    std::cout << "Dumpping to files..."<< std::endl;
    std::string bunch_info = "bunch";
    otpt.dump_to_file(bunch_info);
	std::string cavity_voltage = "cavity";
	cvt.dump_voltage(cavity_voltage);
	std::string t = "tempT";
	std::string pz = "tempPz";
	std::string vc = "tempVc";
	//Dump the results
	/*
	dump_to_file (t,temp_t);
	dump_to_file(pz,temp_pz);
	dump_to_file (vc, temp_vc);
	bm1.bnches[0].dump_coords_to_file(path+std::to_string(0)+std::to_string(0)+"after_coords");
    bm1.bnches[0].dump_to_file(path+std::to_string(0)+std::to_string(0)+"after");
    */
    std::cout << "Finished:)"<< std::endl;

	return 0;
}
