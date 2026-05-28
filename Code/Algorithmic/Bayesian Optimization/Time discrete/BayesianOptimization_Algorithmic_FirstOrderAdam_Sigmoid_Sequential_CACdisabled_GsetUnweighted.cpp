#include <mpi.h>
#include <cmath>
#include <fstream>
#include <bayesopt/bayesopt.hpp>
#include <bayesopt/parameters.hpp>
#include <boost/numeric/ublas/assignment.hpp>
#include <array>
#include <chrono>
#include <random>

void loadJ(std::vector<std::vector<int>>& J, const int N, const int problem, double &E_target, double TTS_treshhold) 
{
    std::string path_Gset = "/dodrio/scratch/projects/starting_2026_056/problem_sets/Gset/";
    std::ifstream file(path_Gset + "G" + std::to_string(problem) + ".mtx");
    if (!file.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return;
    }

    std::string line;
    int E;
    while (std::getline(file, line)) {
        if (line[0] == '%') {
            // Ignore lines starting with '%'
            continue;
        }
        else {
            break; // Exit the loop after reading the line
        }
    }

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int i, j;
        if (!(iss >> i >> j)) {
            std::cerr << "Error parsing line: " << line << std::endl;
            continue;
        }
        if (j - 1 < N) {
            J[j - 1].push_back(i - 1);
            J[i - 1].push_back(j - 1);
        }
    }
    // Calculate E_target
    std::vector<double> G_energies = {11624,11620,11622,11646,11631,2178,2006,2005,2054,2000,564,556,582,3064,3050,3052,3047,992,906,941,931,13359,13344, 13337,13340, 13328,3341,3298,3405,3413,3310,1410,1382,1384,7687,7680,7691,7688,2408,2400,2405,2481,6660,6650,6654,6649,6657,6000,6000,5880,3848,3851,3850,3852,10299,4017,3494};
    E_target = 0.; 
    for (int i = 0; i < N; i++) {
        E_target += static_cast<double>(J[i].size());
    }
    E_target *= 0.5;
    E_target -= 2. * TTS_treshhold*G_energies[problem-1];
    return;
}


std::array<double, 7> Ising_Machine(const double alpha, const double beta, const double theta, const double beta1, const double beta2, const double eta, const double gamma, const double dt, const double alpha_CAC, const double a_CAC, const double delta_CAC, const double rho_CAC, const double ksi_CAC, const double Gamma_CAC, const int n_x_CAC, const int n_e_CAC, const int DT_CAC, const int n_runs, const int n_it, const int N, const int problem){
    
    // Initialization  
    // Coupling matrix
    std::vector<std::vector<int>> J(N);
    double E_target = 0.;
    double E_target_percentage =  99.5 / 100.;
    loadJ(J, N, problem, E_target, E_target_percentage);    
    // Measures
    double TTS_CPU = 0., TTS_Euler = 0., SR_tr = 0., SR = 0., T_a_CPU = 0., T_a_Euler = 0., E_min = std::numeric_limits<double>::max();    
 
    for (int run = 0; run < n_runs; run++)
    {
        // INITIALIZE
        bool target_energy_reached = false;
        // Gaussian white noise and seed
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count()+run;
        std::default_random_engine generator(seed);  
        std::normal_distribution<double> dist(0., std::sqrt(dt));
        // States
        double coupling_term, grad_f_i, t, E; t = dt;
         
        std::vector<double> x(N), v(N), w(N);
        double x_i, x_bin_i, v_i, w_i; 
        
        
        std::vector<double> x_bin(N);                   
        // Initialize states
        for (int i = 0; i < N; ++i)
        {
            x[i] = gamma * dist(generator);
            v[i] = gamma * dist(generator);
            w[i] = std::abs(gamma * dist(generator));
        }
        
        // Start time
        auto start = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration;
        // Euler iterations
        for (int iter = 0; iter < n_it; iter++)
        {
            
            // Update x_i, sum over spins
            for (int i = 0; i < N; i++)
            {
                
                // Calculate coupling term:
                coupling_term = 0.;
                for (int j : J[i])
                {
                    coupling_term += x[j];
                }
                coupling_term *= -beta; // Minus because anti ferromagnetic                    
                // Load previous m,v,x:
                x_i = x[i];
                v_i = v[i];
                w_i = w[i];
                // Non linearity:
                grad_f_i = x_i - std::tanh(alpha * x_i + coupling_term);
                // Update m,v,x,CAC:
                
                v[i] = beta1*v_i + (1.0f-beta1)  * (grad_f_i);
                w[i] = beta2*w_i + (1.0f-beta2) * (std::pow(grad_f_i, 2));
                x[i] = x_i - eta * v[i] / (std::sqrt(w[i]*t) + 0.00000001) + gamma * dist(generator); 
                
            }
            
            
            // Ising binary energy:
            for (int i = 0; i < N; i++){    x_bin[i] = std::copysignf(1.0f, x[i]);   }
            E = 0.;
            for (int i = 0; i < N; i++)
            {
                x_bin_i = x_bin[i];
                for (int j : J[i])
                {
                    E += x_bin_i * x_bin[j];
                }
            }
            E *= 0.5;
            //If solution is found, update TTS
            if (E < E_min)
            {
                E_min = E;
            }
            if (target_energy_reached == false && E <= E_target) 
            {
                // Stop time
                auto end = std::chrono::high_resolution_clock::now();
                duration = end - start;
                T_a_CPU += duration.count();
                T_a_Euler += t;
                SR_tr += 1.;
                       
                target_energy_reached = true;             
            }
            // Time
            t += dt;
        } // End of iteration loop
           
        if (E <= E_target) 
        {
            SR += 1.;                 
        }   
    } // End of run loop
    
    // Compute SR_tr and TTS
    if (SR_tr == 0.)
    {
        TTS_CPU = 0.;
        TTS_Euler = 0.;
    }
    if (SR_tr != 0.)
    {
        T_a_CPU /= SR_tr;
        T_a_Euler /= SR_tr;
    }
    SR_tr /= static_cast<double>(n_runs);
    SR /= static_cast<double>(n_runs);
    if (SR_tr <= 0.99 && SR_tr != 0.)
    {
        TTS_CPU = T_a_CPU * std::log(0.01) / std::log(1. - SR_tr);
        TTS_Euler = T_a_Euler * std::log(0.01) / std::log(1. - SR_tr);
    }
    else if (SR_tr >= 0.99)
    {
        TTS_CPU = T_a_CPU;
        TTS_Euler = T_a_Euler;
    }

    return {TTS_CPU,TTS_Euler,SR_tr,SR,T_a_CPU, T_a_Euler,E_min};
} // End of TTS function definition.

class BayesianOptimization : public bayesopt::ContinuousModel {
    public:
        BayesianOptimization(size_t bayesopt_dim, bayesopt::Parameters params,const int N, const int problem, std::ofstream &outfile)
            : bayesopt::ContinuousModel(bayesopt_dim, params), N(N), problem(problem),outfile(outfile) {}
    
        double evaluateSample( const vectord &Xi ) {
            // Extract variables from the sample vector
            this->alpha = Xi(0); this->beta = Xi(1); this->beta1 = Xi(2); this->beta2 = Xi(3); this->eta = Xi(4); this->gamma = Xi(5); 

            std::array<double, 7> results = Ising_Machine(this->alpha, this->beta, this->theta, this->beta1, this->beta2, this->eta, std::pow(10.,this->gamma), std::pow(10.,this->dt), this->alpha_CAC, this->a_CAC, this->delta_CAC, this->rho_CAC, this->ksi_CAC, this->Gamma_CAC, this->n_x_CAC, this->n_e_CAC, this->DT_CAC, this->n_runs, this->n_it, this->N, this->problem);
            double TTS_CPU,TTS_Euler,SR_tr,SR,T_a_CPU,T_a_Euler,E_min;
            TTS_CPU = results[0]; 
            TTS_Euler = results[1];
            SR_tr = results[2];
            SR = results[3];
            T_a_CPU = results[4];
            T_a_Euler = results[5];
            E_min = results[6];

            outfile << "alpha = " << alpha << " , beta = " << beta << " , theta = " << theta << " , beta1 = " << beta1 << " , beta2 = " << beta2 << " , eta = " << eta << " , log_10_gamma = " << gamma << " , log_10_dt = " << dt << " , alpha_CAC = " << alpha_CAC << " , a_CAC = " << a_CAC << " , delta_CAC = " << delta_CAC << " , rho_CAC = " << rho_CAC << " , ksi_CAC = " << ksi_CAC << " , Gamma_CAC = " << Gamma_CAC << " , n_x_CAC = " << n_x_CAC << " , n_e_CAC = " << n_e_CAC << " , DT_CAC = " << DT_CAC << std::endl;
            outfile << "TTS_CPU = " << TTS_CPU << " , TTS_Euler = " << TTS_Euler << " , SR_tr = " << SR_tr << " , SR = " << SR << " , T_a_CPU = " << T_a_CPU << " , T_a_Euler = " << T_a_Euler << " , E_min = " << E_min << std::endl << std::endl;

            double result = TTS_CPU;
            
            if(result != 0.){ result = -1./result; }

            return result;
        }
    
    private:
        double alpha = 0.5;
        double beta = 0.5;
        double theta = 0.05;
        double beta1 = 0.1;
        double beta2 = 0.2;
        double eta = 1.;
        double gamma = -5.;
        double dt = 0.;

        double alpha_CAC = 3.8;
        double a_CAC = 3.5;
        double delta_CAC = 10.;
        double rho_CAC = 8.;
        double ksi_CAC = 0.004;
        double Gamma_CAC = 0.004;
        int n_x_CAC = 10;
        int n_e_CAC = 10;
        int DT_CAC = 400;

        double TTS_treshhold = 99.5 / 100.;
        int n_runs = 100;
        int n_it = 15000;

        const int N, problem; 
        std::ofstream &outfile; 
    public:
        double alpha_min = -2., alpha_max = 2.;
        double beta_min = 0., beta_max = 3.;
        double theta_min = -200., theta_max = 1.;
        double beta1_min = -200., beta1_max = 1.;
        double beta2_min = -200., beta2_max = 1.;
        double eta_min = 1., eta_max = 200.;
        double gamma_min = -10., gamma_max = 2.;
        double dt_min = -3., dt_max = 2.;

        double alpha_CAC_min = 0., alpha_CAC_max = 5.;
        double a_CAC_min = 0.5, a_CAC_max = 5.;
        double delta_CAC_min = 1., delta_CAC_max = 20.;
        double rho_CAC_min = 0., rho_CAC_max = 10.;
        double ksi_CAC_min = 0., ksi_CAC_max = 0.01;
        double Gamma_CAC_min = 0., Gamma_CAC_max = 0.01;
        int n_x_CAC_min = 1, n_x_CAC_max = 20;
        int n_e_CAC_min = 1, n_e_CAC_max = 20;
        int DT_CAC_min = 10, DT_CAC_max = 1000;

   };//End of optimization class definition.



    
int main(int argc, char* argv[])
{
    //Initialize MPI parallellisation:
    MPI_Init(&argc, &argv);

    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  

    // Problem constants
    const std::vector<int> N_s = { 800,800,800,800,800,800,800,800,800,2000,2000,2000,2000,2000,2000,2000,2000,2000,1000,1000,1000,1000,1000,1000,1000,1000,1000 };
    const std::vector<int> problems = { 1,2,3,4,5,14,15,16,17,22,23,24,25,26,35,36,37,38,43,44,45,46,47,51,52,53,54 };
    const int N = N_s[rank];
    const int problem = problems[rank];
    // Generate a unique filename for each process
    std::ostringstream filename;
    filename << "BayesianOptimization_Algorithmic_FirstOrderAdam_Sigmoid_Sequential_CACdisabled_" << "G" << problem << ".txt";
    // Extract the string from the ostringstream
    std::string path = "/dodrio/scratch/projects/starting_2026_056/results/run1/";
    std::string full_filename = path + filename.str();
    // Open the file using the full filename
    std::ofstream outfile(full_filename.c_str(), std::ios::app);
    if (!outfile) {
        std::cerr << "Error opening file for writing: " << filename.str() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

////////////////////////////////////////////////////////////// CONSTANTS //////////////////////////////////////////////////////////////////////////
    // BayesOpt model setup
    bayesopt::Parameters params = initialize_parameters_to_default(); 
// BayesOpt parameters  
    params.n_init_samples = 200;   
    params.n_iterations = 600;
    params.noise = std::pow(10.,-3);
    params.verbose_level = 5;
    const size_t bayesopt_dim_ = 6;
    // Set the kernel to a combination of Matern and Rational Quadratic
    params.kernel.name = "kSum(kMaternARD3,kRQISO)";    
    // Use a Student's t Process with Normal-Inverse-Gamma priors
    params.surr_name = "sStudentTProcessNIG";    
    // Use a constant mean function
    params.mean.name = "mConst";
    // Enable learning all params, not only kernels
    params.l_all = true;
////////////////////////////////////////////////////////////// CONSTANTS //////////////////////////////////////////////////////////////////////////
    // Construct the Bayesian optimization object
    BayesianOptimization opt(bayesopt_dim_, params, N, problem, outfile);
    // Set the bounds for each parameter
    boost::numeric::ublas::vector<double> lb(bayesopt_dim_), ub(bayesopt_dim_);
    lb(0) = opt.alpha_min + 0.000000001; lb(1) = opt.beta_min + 0.000000001; lb(2) = opt.beta1_min + 0.000000001; lb(3) = opt.beta2_min + 0.000000001; lb(4) = opt.eta_min + 0.000000001; lb(5) = opt.gamma_min; 
    ub(0) = opt.alpha_max - 0.000000001; ub(1) = opt.beta_max - 0.000000001; ub(2) = opt.beta1_max - 0.000000001; ub(3) = opt.beta2_max - 0.000000001; ub(4) = opt.eta_max - 0.000000001; ub(5) = opt.gamma_max; 
    opt.setBoundingBox(lb, ub);
    // Perform the optimization
    vectord result(bayesopt_dim_);
    opt.optimize(result);  

    // Close the file
    outfile.close();

    MPI_Finalize();
    return 0;
}