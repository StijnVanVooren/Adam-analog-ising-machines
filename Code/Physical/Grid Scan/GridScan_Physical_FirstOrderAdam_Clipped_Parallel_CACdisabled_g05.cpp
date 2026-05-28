#include <mpi.h>
#include <fstream>
#include <cmath>
#include <array>
#include <chrono>

void loadJ(std::vector<std::vector<int>>& J, const int N, const int problem, double &E_target, double TTS_treshhold) 
{
    std::string path_biq_mac = "/dodrio/scratch/projects/starting_2026_056/problem_sets/g05/";
    //Load J from file:
    std::string filename = "g05_" + std::to_string(N) + "." + std::to_string(problem);
    std::ifstream file(path_biq_mac + filename);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    std::string line;
    // Ignore the first line
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int i, j;
        double k;
        if (!(iss >> i >> j >> k)) {
            std::cerr << "Error parsing line: " << line << std::endl;
            continue;
        }

        // Adjust index if necessary
        i--; j--;
        if (k == 1) {
            J[i].push_back(j);
            J[j].push_back(i); //Symmetrisch
        }
        else if (k != 0) {
            std::cerr << "Error: Non-zero value not equal to 1 encountered in coupling matrix while loading." << std::endl;
        }
    }
    file.close();
    //Calculate E_bigmac
    std::vector<double> biqmac_energies60 = { 536.,532.,529.,538.,527.,533.,531.,535.,530.,533. };
    std::vector<double> biqmac_energies80 = { 929., 941., 934., 923., 932., 926., 929., 929., 925., 923. };
    std::vector<double> biqmac_energies100 = { 1430., 1425., 1432., 1424., 1440., 1436., 1434., 1431., 1432., 1430. };
    E_target = 0.;
    for (int i = 0; i < N; i++)
    {
        E_target += static_cast<double>(J[i].size());
    }
    E_target *= 0.5;
    if (N == 60) { E_target -= 2. * TTS_treshhold*biqmac_energies60[problem]; }
    if (N == 80) { E_target -= 2. * TTS_treshhold*biqmac_energies80[problem]; }
    if (N == 100){ E_target -= 2. * TTS_treshhold*biqmac_energies100[problem];}

    return;
}


std::array<double, 7> Ising_Machine(const double alpha, const double beta, const double theta, const double beta1, const double beta2, const double eta, const double gamma, const double dt, const double alpha_CAC, const double a_CAC, const double delta_CAC, const double rho_CAC, const double ksi_CAC, const double Gamma_CAC, const int n_x_CAC, const int n_e_CAC, const int DT_CAC, const int n_runs, const int n_it, const int N, const int problem){
    
    // Initialization  
    // Coupling matrix
    std::vector<std::vector<int>> J(N); 
    double E_target = 0.;
    double E_target_percentage =  100. / 100.;
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
        std::vector<double> x_new(N), v_new(N), w_new(N);
        double x_i, x_bin_i, v_i, w_i; 
         
        double x_tmp;
        
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
                grad_f_i = (1.f-alpha)*x_i - coupling_term;
                // Update m,v,x,CAC:
                v_new[i] = v_i + dt * (1.0f-beta1)  * (grad_f_i - v_i);
                w_new[i] = w_i + dt * (1.0f-beta2) * (std::pow(grad_f_i, 2) - w_i);
                x_new[i] = x_i - dt * eta * v_i / (std::sqrt(w_i*t) + 0.00000001) + gamma * dist(generator); 
                
                
            }
            for(int i=0; i<N; i++) { if(std::abs(x_new[i])<=0.4){ x[i] = x_new[i]; }  } 
            v = v_new;
            w = w_new;
            
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
                break;       
                target_energy_reached = true;             
            }
            // Time
            t += dt;
        } // End of iteration loop
           
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


double alpha = 0.5;
double beta = 0.5;
double theta = 0.05;
double beta1 = 0.99;
double beta2 = 0.99;
double eta = 1.;
double gamma = -5.;
double dt = -2.;

double alpha_CAC = 3.8;
double a_CAC = 3.5;
double delta_CAC = 10.;
double rho_CAC = 8.;
double ksi_CAC = 0.004;
double Gamma_CAC = 0.004;
int n_x_CAC = 10;
int n_e_CAC = 10;
int DT_CAC = 400;

double TTS_treshhold = 100. / 100.f;
int n_runs = 400;
int n_it = 10000;

int N, problem; 
std::ofstream &outfile; 

double alpha_min = -3., alpha_max = 3.;
double beta_min = 0., beta_max = 4.;
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

double best_value = <<<best_value_init>>>;
double best_alpha = -1., best_beta = -1., best_theta = -1., best_beta1 = -1., best_beta2 = -1., best_eta = -1., best_gamma = -1., best_dt = -1., best_alpha_CAC = -1., best_a_CAC = -1., best_delta_CAC = -1., best_rho_CAC = -1., best_ksi_CAC = -1., best_Gamma_CAC = -1., best_n_x_CAC = -1., best_n_e_CAC = -1., best_DT_CAC = -1.;

        
double TTS_treshhold = 100. / 100.f; 
int n_runs = 400;
int n_it = 10000;
int resolution = 30;