#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <parallel/numeric>
#include <parallel/algorithm>


void dump_to_file(std::string& path, std::vector<double>& data){
    std::filebuf fb;
	fb.open(path, std::ios::out);
	std::ostream os(&fb);
	for (int i = 0; i<data.size(); ++i) {
		os << data[i] << '\n';
	}
	fb.close();
};


double covariance(std::vector<double>& a, std::vector<double>& b, double meanA, double meanB){
    std::vector<double> temp(a.size(),0);
#pragma omp parallel for
    for(int i = 0;i<a.size();++i){
        temp[i] = (a[i]-meanA)*(b[i]-meanB);
    }
    return __gnu_parallel::accumulate(temp.begin(),temp.end(),0.0)/a.size();
}


double max_para(std::vector<double>& x){
    return x[std::distance(x.begin(),__gnu_parallel::max_element(x.begin(),x.end()))];
}

double min_para(std::vector<double>& x){
    return x[std::distance(x.begin(),__gnu_parallel::min_element(x.begin(),x.end()))];
}

// the function do  the binning of the particles.
// "data" is the vector of input data
// "counts" is the vector stores the number of particles in each bin.
// "bin_index" is the vector stores the index of bin where ith particle is located.
// "bin_size" is the size(width) of the bin.
// "n_bin_each_side" is the number of bins on each side from center of the bunch.
void binning(std::vector<double>& data, std::vector<unsigned int>& counts, std::vector<unsigned int>& bin_index, double data_mean, double bin_size, double n_bins_each_side){
    double bin_max = data_mean+bin_size*n_bins_each_side; // center of the right most bin,
    double bin_min = data_mean-bin_size*n_bins_each_side; // center of the left most bin,
    unsigned int tot_bins = 2*n_bins_each_side+1;
    double left = bin_min-data_mean-bin_size*0.5; // left edge of the current range where the particle is located,
    double right = bin_max+bin_size*0.5; // rigth edge of the current range where the particle is located.
    for(int i = 0;i<data.size();++i){
        double left = data_mean-bin_size*0.5; // left edge of the current range where the particle is located,
        double right = bin_max+bin_size*0.5; // rigth edge of the current range where the particle is located.
        while(right-left>bin_size){
            if(data[i]>(left+right)*0.5){
                left = (left+right-bin_size)*0.5;
            }
            else{
                right = (left+right+bin_size)*0.5;
            }
        }
        bin_index[i] = n_bins_each_side+int((left+right)*0.5/bin_size);
        counts[bin_index[i]] += 1;
    }
}
