import os
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import re
from itertools import product
from tkinter import ttk
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
import math
from matplotlib.ticker import FuncFormatter

from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Affine2D, IdentityTransform


import sys
import subprocess


DEFAULT_PATH = r"/Users/stijnvanvooren/Documents/PhD/Adam/Results/Adam_Final/"
BENCHMARK_PATH = r"/Users/stijnvanvooren/Documents/PhD/Adam/Benchmarksets/"

#DEFAULT_PATH = r"/Users/stijnvanvooren/Documents/PhD/Adam/Results/"
#BENCHMARK_PATH = r"/Users/stijnvanvooren/Documents/PhD/Adam/Benchmarksets/"

OPTION_KEYS = [
    "Hyper_Parameter_Scan", "Implementation", "Optimization_Type","Non_linearity",
    "Update_type", "Chaotic_Amplitude_Control", "Problem_Set"
]

OPTION_KEYWORDS = {
    "Hyper_Parameter_Scan": ["GridScan", "BayesianOptimization"],
    "Implementation": ["Physical", "Algorithmic"],
    "Optimization_Type": ["FirstOrderAdam", "Adam", "Momentum", "GradientDescent" ],
    "Update_type": ["Sequential", "Parallel"],
    "Non_linearity": ["Polynomial", "Sigmoid", "Periodic", "Clipped"],
    "Chaotic_Amplitude_Control": [
        "CACauto", "CACphys", "CACalgo",
        "CFCauto", "CFCphys", "CFCalgo", "CACdisabled"
    ],
    "Problem_Set": ["g05", "GsetWeighted","GsetUnweighted"]
}

PRIMARY_PARAMS = [
    "None", "alpha", "beta", "theta", "beta1", "beta2", "eta", 
    "log_10_gamma", "log_10_dt", "alpha_CAC", "a_CAC", "delta_CAC", 
    "rho_CAC", "ksi_CAC", "Gamma_CAC", "n_x_CAC", "n_e_CAC", "DT_CAC"
]

PERFORMANCE_METRICS = [
    "TTS_CPU", "TTS_Euler", "TTS_Euler_eta_corrected", "SR_tr", "SR", "T_a_CPU", "T_a_Euler", "E_min", "E_min_original"
]

BEST_PARAMS = [
    "best_alpha", "best_beta", "best_theta", "best_beta1", "best_beta2", "best_eta",
    "best_gamma", "best_dt", "best_alpha_CAC", "best_a_CAC", "best_delta_CAC",
    "best_rho_CAC", "best_ksi_CAC", "best_Gamma_CAC", "best_n_x_CAC", "best_n_e_CAC", "best_DT_CAC",
    "best_value"
]

groups = [
    ("Primary Parameters", PRIMARY_PARAMS),
    ("Performance Metrics", PERFORMANCE_METRICS),
    ("Best Parameters", BEST_PARAMS),
]

RAW2BEST = {
    "alpha":          "best_alpha",
    "beta":           "best_beta",
    "theta":          "best_theta",
    "beta1":          "best_beta1",
    "beta2":          "best_beta2",
    "eta":            "best_eta",
    "log_10_gamma":   "best_gamma",
    "log_10_dt":      "best_dt",
    "alpha_CAC":      "best_alpha_CAC",
    "a_CAC":          "best_a_CAC",
    "delta_CAC":      "best_delta_CAC",
    "rho_CAC":        "best_rho_CAC",
    "ksi_CAC":        "best_ksi_CAC",
    "Gamma_CAC":      "best_Gamma_CAC",
    "n_x_CAC":        "best_n_x_CAC",
    "n_e_CAC":        "best_n_e_CAC",
    "DT_CAC":         "best_DT_CAC",
}

# Constants
G05_N_VALUES = [60, 80, 100]
G05_PROBLEMS = list(range(10))
GSET_UNWEIGHTED = [1, 2, 3, 4, 5, 14, 15, 16, 17, 22, 23, 24, 25, 26, 35, 36, 37, 38, 43, 44, 45, 46, 47, 51, 52, 53, 54]
GSET_WEIGHTED = [6, 7, 8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 27, 28, 29, 30, 31, 32, 33, 34, 39, 40, 41, 42]

BEST_KNOWN_VALUES_g05_60 = [536.,532.,529.,538.,527.,533.,531.,535.,530.,533.]
BEST_KNOWN_VALUES_g05_80 = [929., 941., 934., 923., 932., 926., 929., 929., 925., 923.]
BEST_KNOWN_VALUES_g05_100 = [1430., 1425., 1432., 1424., 1440., 1436., 1434., 1431., 1432., 1430.]
BEST_KNOWN_VALUES_GSET = [11624,11620,11622,11646,11631,2178,2006,2005,2054,2000,564,556,582,3064,3050,3052,3047,992,906,941,931,13359,13344, 13337,13340, 13328,3341,3298,3405,3413,3310,1410,1382,1384,7687,7680,7691,7688,2408,2400,2405,2481,6660,6650,6654,6649,6657,6000,6000,5880,3848,3851,3850,3852,10299,4017,3494]


class Data:
    def __init__(self, path, filename):
        print()
        self.filename = filename
        self.path = path
        self.filepath = os.path.join(path, filename)
        self.samples = []        # List of dicts
        self.best_parameters = {}
        self.best_measures = {}
        for metric in PERFORMANCE_METRICS:
            if metric in ["SR_tr", "SR","E_min_original"]:
                self.best_measures[metric] = -math.inf  # maximize these
            else:
                self.best_measures[metric] = math.inf   # minimize these

        self.best_value = math.inf

        # Remove file extension and split by '_'
        parts = filename.replace(".txt", "").split("_")

        # First 5 parts are fixed options
        self.options = {
            key: parts[i] for i, key in enumerate(OPTION_KEYS[:-1])  # exclude Problem_Set for now
        }
        
        print(f"Options parsed from filename: {self.options}")

        # The rest is the problem string
        self.problem = "_".join(parts[6:])
        print(f"Problem parsed from filename: {self.problem}")

        if self.problem.startswith("g05"):
            # Replace last underscore with dot
            problem_parts = self.problem.rsplit('_', 1)
            if len(problem_parts) == 2:
                self.problem_file = BENCHMARK_PATH+"g05/"+problem_parts[0] + '.' + problem_parts[1]
                self.problem_number = int(problem_parts[1])
            else:
                print("Warning: Problem string does not have an underscore to replace.")
            problem_set = "g05"
        elif self.problem.startswith("G"):
            self.problem_file = BENCHMARK_PATH+"Gset/"+self.problem + ".mtx"
            rest = self.problem[1:]
            if rest.isdigit():
                num = int(rest)
                if num in GSET_UNWEIGHTED:
                    problem_set = "GsetUnweighted"
                elif num in GSET_WEIGHTED:
                    problem_set = "GsetWeighted"
                else:
                    problem_set = "UnknownGset"
                self.problem_number = int(rest)
            else:
                problem_set = "UnknownGset"
        else:
            problem_set = "Unknown"


        # Add to options dict
        self.options["Problem_Set"] = problem_set


        print(f"Loaded: {filename}, problem={self.problem}, problem number={self.problem_number}, options={self.options}")
        self.read_problem()
        self._load()
        print()

    def _parse_line_to_dict(self, line):
        """Parses a single line like 'a = 1 , b = 2' into a dict."""
        return {
            key.strip(): self._convert_value(value.strip())
            for key, value in re.findall(r"(\w+)\s*=\s*([-\w.+eE]+)", line)

        }

    def _convert_value(self, val):
        try:
            if 'e' in val or '.' in val or '-' in val:
                return float(val)
            return int(val)
        except ValueError:
            return val
        
    def read_problem(self):
        found_N = False  # flag to check if N is found
        w = None
        with open(self.problem_file, 'r') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('%'):
                    continue

                if not found_N:
                    # First non-comment line: get N
                    self.N = int(line.split()[0])
                    w = np.zeros((self.N, self.N))
                    found_N = True
                else:
                    # After N is found, sum weights
                    parts = line.split()
                    if len(parts) == 2:
                        w[int(parts[0])-1][int(parts[1])-1] = 1
                        w[int(parts[1])-1][int(parts[0])-1] = 1
                    elif len(parts) == 3:
                        w[int(parts[0])-1][int(parts[1])-1] = float(parts[2])
                        w[int(parts[1])-1][int(parts[0])-1] = float(parts[2])
        self.sum_w_ij = np.sum(w)
        print(self.sum_w_ij)
        if self.options["Problem_Set"] == "g05":
            if self.N == 60:
                self.E_original_best_known = BEST_KNOWN_VALUES_g05_60[self.problem_number]
            elif self.N == 80:
                self.E_original_best_known = BEST_KNOWN_VALUES_g05_80[self.problem_number]
            elif self.N == 100:
                self.E_original_best_known = BEST_KNOWN_VALUES_g05_100[self.problem_number]
            else:
                print(f"Unknown N value for g05: {self.N}")
        elif self.options["Problem_Set"] == "GsetWeighted":
            self.E_original_best_known = BEST_KNOWN_VALUES_GSET[self.problem_number-1]#Indexing starts at 0, so we subtract 1
        elif self.options["Problem_Set"] == "GsetUnweighted":
            self.E_original_best_known = BEST_KNOWN_VALUES_GSET[self.problem_number-1]#Indexing starts at 0, so we subtract 1
            
        self.E_best_known = self.sum_w_ij/2 - 2*self.E_original_best_known
        print(f"Problem {self.problem} has N = {self.N} and sum_w_ij = {self.sum_w_ij}, E_original_best_known = {self.E_original_best_known}, E_best_known = {self.E_best_known}")

    def _load(self):
        # Determine which TTS metric to use for best parameter selection
        tts_metric_key = None
        if "Physical" in self.filename:
            tts_metric_key = "TTS_Euler"
        elif "Algorithmic" in self.filename:
            tts_metric_key = "TTS_CPU"

        with open(self.filepath, "r") as f:
            content = f.read()

        blocks = [block.strip() for block in content.split("\n\n") if block.strip()]

        for block in blocks:
            lines = block.splitlines()
            # Extract all samples first
            sample_data = {}
            for line in lines:
                sample_data.update(self._parse_line_to_dict(line))
            #if sample_data["E_min_original"] exists (is loaded in):
            if "E_min_original" not in sample_data and "E_min" in sample_data:
                sample_data["E_min_original"] = self.sum_w_ij/4 - float(sample_data["E_min"])/2

            if sample_data:
                self.samples.append(sample_data)

            if tts_metric_key and tts_metric_key in sample_data:
                try:
                    current_tts_value = float(sample_data[tts_metric_key])
                except ValueError:
                    print(f"Warning: Invalid value for {tts_metric_key} in sample: {sample_data[tts_metric_key]}")
                    current_tts_value = None
            else:
                print(f"Warning: {tts_metric_key} not found in sample data: {sample_data}")
                current_tts_value = None

            if current_tts_value is not None and current_tts_value <= self.best_value and current_tts_value != 0:
                self.best_value = current_tts_value

                # build a fresh dict each time we find an equally‑good or better candidate
                bp = {"best_value": current_tts_value}

                for raw_key, best_key in RAW2BEST.items():
                    if raw_key in sample_data:
                        bp[best_key] = sample_data[raw_key]

                self.best_parameters = bp

            # Still update all performance metrics
            for key, value in sample_data.items():
                if key in PERFORMANCE_METRICS:
                    try:
                        value = float(value)
                        if key in ["SR_tr", "SR", "E_min_original"]:  # maximize
                            if value > self.best_measures[key]:
                                self.best_measures[key] = value
                        else:  # minimize
                            if value < self.best_measures[key] and value != 0:
                                self.best_measures[key] = value
                    except ValueError:
                        continue

            # TTS_Euler_eta_corrected:
            if "TTS_Euler" in sample_data:
                try:
                    eta_corr = float(sample_data.get("eta", 1.0))

                    if self.options["Optimization_Type"] == "FirstOrderAdam":
                        beta1 = float(sample_data["beta1"])
                        beta2 = float(sample_data["beta2"])

                        absorbed_factor = np.sqrt(1.0 - beta2) / (1.0 - beta1)
                        absorbed_factor = 1
                        eta_corr = eta_corr / absorbed_factor

                    value = float(sample_data["TTS_Euler"]) * eta_corr
                    sample_data["TTS_Euler_eta_corrected"] = value

                    if value < self.best_measures["TTS_Euler_eta_corrected"] and value != 0:
                        self.best_measures["TTS_Euler_eta_corrected"] = value

                except (KeyError, ValueError, ZeroDivisionError):
                    pass
    

        if self.best_measures["E_min_original"] != self.sum_w_ij/4 - self.best_measures["E_min"]/2:
            print(f"Warning: E_min_original ({self.best_measures['E_min_original']}) does not match sum_w_ij/4 - E_min/2 ({self.sum_w_ij/4 - self.best_measures['E_min']/2})")


        #print best measures
        #for key, value in self.best_measures.items():
            #if key in ["SR_tr", "SR"]:
                #print(f"Best {key} = {value}")
            #else:
                #print(f"Best {key} = {value:.2e}")
        #print best parameters
        #print(f"Best parameters for {self.problem}:")
        #for key, value in self.best_parameters.items():
        #    print(f"{key} = {value}")
        

        

    def print_summary(self):
        print(f"\n📄 Loaded file: {self.filepath}")
        print(f"🔢 Number of samples: {len(self.samples)}")

        if self.samples:
            print(f"📋 Keys in samples: {', '.join(self.samples[0].keys())}")
            print(f"\n🔍 Sample 0:")
            for k, v in self.samples[0].items():
                print(f"  {k} = {v}")
        if self.samples:
            print(f"📋 Keys in samples: {', '.join(self.samples[3].keys())}")
            print(f"\n🔍 Sample 3:")
            for k, v in self.samples[3].items():
                print(f"  {k} = {v}")

        if self.best_parameters:
            print(f"\n🏆 Best parameters:")
            for k, v in self.best_parameters.items():
                print(f"  {k} = {v}")

        if self.best_value is not None:
            print(f"\n⭐ Best value = {self.best_value}")

    def __repr__(self):
        return f"<Data: {len(self.samples)} samples, best_value={self.best_value}>"




class SimulationDataGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulation Data GUI")
        self.option_values = {key: set() for key in OPTION_KEYS}
        self.check_vars = {key: {} for key in OPTION_KEYS}
        self.cpp_files = []
        self.loaded_data = []
        # Make sure this line comes first
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True)
        self.scatter_tab = ttk.Frame(self.notebook)
        self.scatter_tab.pack(fill='both', expand=True)
        self.scatter_x_var = tk.StringVar()
        self.scatter_y_var = tk.StringVar()

        # Now you can create tabs
        self.create_scatter_tab()

        self.barplot_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.barplot_tab, text="Barplot")
        self.setup_barplot_tab()

        self.latex_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.latex_tab, text="LaTeX")
        self.setup_latex_tab()


        # Layout
        tk.Button(root, text="Select Folder", command=self.select_folder).pack(pady=10)
        self.check_frame = tk.Frame(root)
        self.check_frame.pack(pady=5)
        
        tk.Button(root, text="Load data files", command=self.load).pack(pady=10)

        tk.Button(root, text="Print Selected Options", command=self.print_selected).pack()
        

    def create_scatter_tab(self):
        self.notebook.add(self.scatter_tab, text="Scatter Plot")

        title = ttk.Label(self.scatter_tab, text="Scatter Plot: Select X and Y Variables", font=("Helvetica", 14, "bold"))
        title.pack(pady=10)

        selector_frame = ttk.Frame(self.scatter_tab)
        selector_frame.pack(pady=5, padx=10)

        # Get list of available variables from the first loaded data (assumes consistent keys)
        available_vars = set()
        for data_obj in self.loaded_data:
            for sample in data_obj.samples:
                available_vars.update(sample.keys())
            break  # just use the first to get keys

        sorted_vars = sorted(available_vars)

        self.scatter_x_var = tk.StringVar(value="None")
        self.scatter_y_var = tk.StringVar(value=sorted_vars[1] if len(sorted_vars) > 1 else "None")

        # X Variable
        x_frame = ttk.LabelFrame(selector_frame, text="X Variable")
        x_frame.pack(side='left', padx=10)
        for var in sorted_vars:
            ttk.Radiobutton(x_frame, text=var, variable=self.scatter_x_var, value=var).pack(anchor='w')

        # Y Variable
        y_frame = ttk.LabelFrame(selector_frame, text="Y Variable")
        y_frame.pack(side='left', padx=10)
        for var in sorted_vars:
            ttk.Radiobutton(y_frame, text=var, variable=self.scatter_y_var, value=var).pack(anchor='w')

        # Plot Button
        plot_btn = ttk.Button(self.scatter_tab, text="Plot Scatter", command=self.plot_scatter)
        plot_btn.pack(pady=10)

    def update_scatter_tab(self):
        # Clear previous widgets in scatter_tab
        for widget in self.scatter_tab.winfo_children():
            widget.destroy()

        title = ttk.Label(self.scatter_tab, text="Scatter Plot: Select X and Y Variables", font=("Helvetica", 14, "bold"))
        title.pack(pady=10)

        selector_frame = ttk.Frame(self.scatter_tab)
        selector_frame.pack(pady=5, padx=10)



        groups = [
            ("Primary Parameters", PRIMARY_PARAMS),
            ("Performance Metrics", PERFORMANCE_METRICS),
            ("Best Parameters", BEST_PARAMS),
        ]

        # Initialize variables if not existing
        if not hasattr(self, "scatter_x_var"):
            self.scatter_x_var = tk.StringVar(value="None")
        if not hasattr(self, "scatter_y_var"):
            self.scatter_y_var = tk.StringVar(value="None")

        def create_group(frame, var, axis_label):
            axis_frame = ttk.Frame(frame)
            axis_frame.pack(side="left", padx=10)
            axis_title = ttk.Label(axis_frame, text=axis_label, font=("Helvetica", 12, "bold"))
            axis_title.pack()

            groups_frame = ttk.Frame(axis_frame)
            groups_frame.pack()

            for col, (group_name, var_list) in enumerate(groups):
                group_frame = ttk.LabelFrame(groups_frame, text=group_name)
                group_frame.grid(row=0, column=col, padx=5, sticky="n")


                for v in var_list:
                    rb = ttk.Radiobutton(group_frame, text=v, variable=var, value=v)
                    rb.pack(anchor="w")

        create_group(selector_frame, self.scatter_x_var, "X Variable")
        create_group(selector_frame, self.scatter_y_var, "Y Variable")

        plot_btn = ttk.Button(self.scatter_tab, text="Plot Scatter", command=self.plot_scatter)
        plot_btn.pack(pady=10)

    def setup_barplot_tab(self):
        # Clear previous widgets
        for widget in self.barplot_tab.winfo_children():
            widget.destroy()

        title = ttk.Label(self.barplot_tab, text="Barplot Generator", font=("Helvetica", 14, "bold"))
        title.pack(pady=10)

        Problem_Set = OPTION_KEYWORDS["Problem_Set"]

        # Measure selection (Radiobuttons)
        self.barplot_measure_var = tk.StringVar(value=PERFORMANCE_METRICS[0])  # default selection

        measure_frame = ttk.LabelFrame(self.barplot_tab, text="Select Measure")
        measure_frame.pack(pady=5, fill="x", padx=10)

        for metric in PERFORMANCE_METRICS:
            rb = ttk.Radiobutton(
                measure_frame, text=metric,
                variable=self.barplot_measure_var, value=metric
            )
            rb.pack(anchor="w")

        # Benchmark Set selection (Radiobuttons)
        self.barplot_problem_set_var = tk.StringVar(value=Problem_Set[0])  # default selection

        benchmark_set_frame = ttk.LabelFrame(self.barplot_tab, text="Select Benchmark Set")
        benchmark_set_frame.pack(pady=5, fill="x", padx=10)

        for problem_set in Problem_Set:
            rb = ttk.Radiobutton(
                benchmark_set_frame, text=problem_set,
                variable=self.barplot_problem_set_var, value=problem_set
            )
            rb.pack(anchor="w")

        # Log scale toggle (Checkbox)
        self.barplot_log_var = tk.BooleanVar(value=True)
        log_check = ttk.Checkbutton(
            self.barplot_tab, text="Logarithmic Scale",
            variable=self.barplot_log_var
        )
        log_check.pack(pady=5)

        self.barplot_normalize = tk.BooleanVar(value=True)
        log_check = ttk.Checkbutton(
            self.barplot_tab, text="Normalize Values",
            variable=self.barplot_normalize
        )
        log_check.pack(pady=5)

        # --- Save Path ---
        save_path_frame = ttk.Frame(self.barplot_tab)
        save_path_frame.pack(fill="x", pady=2)
        ttk.Label(save_path_frame, text="Save Path:").pack(side="left", padx=(0, 5))
        #self.barplot_save_path_var = tk.StringVar(value=r"/Users/stijnvanvooren/Documents/PhD/Adam/Results/plots/")
        self.barplot_save_path_var = tk.StringVar(value=DEFAULT_PATH + "plots/")
        ttk.Entry(save_path_frame, textvariable=self.barplot_save_path_var, width=30).pack(side="left", fill="x", expand=True)

        # --- Barplot Name ---
        barplot_name_frame = ttk.Frame(self.barplot_tab)
        barplot_name_frame.pack(fill="x", pady=2)
        ttk.Label(barplot_name_frame, text="Barplot Name:").pack(side="left", padx=(0, 5))
        self.barplot_name_var = tk.StringVar(value="my_barplot")
        ttk.Entry(barplot_name_frame, textvariable=self.barplot_name_var, width=30).pack(side="left", fill="x", expand=True)

        # Generate Button
        barplot_btn = ttk.Button(self.barplot_tab, text="Generate Barplot style 1 (Old Stijn)", command=self.barplot)
        barplot_btn.pack(pady=10)
       
        barplot_btn = ttk.Button(self.barplot_tab, text="Generate Barplot style 2 (New Leen)", command=self.barplot_new)
        barplot_btn.pack(pady=10)

    def setup_latex_tab(self):
        tk.Button(self.latex_tab, text="Export Gset LaTeX table", command=self.export_gset_latex).pack(pady=6)




    def get_all_values(self, data_obj, var_name):
        values = []
        for sample in data_obj.samples:
            if var_name in sample:
                values.append(sample[var_name])
        return values
    
    def get_best_value(self, data_obj):
        if data_obj.best_value is not None:
            return data_obj.best_value
        else:
            return None
    def get_best_parameters(self, data_obj, var_name):
        if data_obj.best_parameters:
            print()
            return float(data_obj.best_parameters.get(var_name))
        else:
            print(f"No best parameters found for {data_obj.problem_file}")
            return None

    def plot_scatter(self):
        x_var = self.scatter_x_var.get()
        y_var = self.scatter_y_var.get()
        print(f"X variable: {x_var}, Y variable: {y_var}")
        if y_var == "None":
            tk.messagebox.showerror("Selection Error", "You must select a Y variable.")
            return
        x_vals = []
        y_vals = []

        if (self.get_group_for_var(x_var) == "Primary Parameters" or self.get_group_for_var(x_var) == "Performance Metrics" or x_var == "None") and (self.get_group_for_var(y_var) == "Primary Parameters" or self.get_group_for_var(y_var) == "Performance Metrics"):
            for data_obj in self.loaded_data:
                y_list = self.get_all_values(data_obj, y_var)
                if not y_list:
                    continue

                if x_var == "None":
                    x_list = np.zeros(len(y_list))
                else:
                    x_list = self.get_all_values(data_obj, x_var)
                    if not x_list:
                        continue

                min_len = min(len(x_list), len(y_list))
                x_vals.extend(x_list[:min_len])
                y_vals.extend(y_list[:min_len])
        else:
            for data_obj in self.loaded_data:
                print(f"Best parameters for {data_obj.filename}: {data_obj.best_parameters}")
                y = self.get_best_parameters(data_obj, y_var)
                if not y:
                    continue

                if x_var == "None":
                    x = 0
                else:
                    x = self.get_best_parameters(data_obj, x_var)
                    if not x:
                        continue

                x_vals.append(x)
                y_vals.append(y)
            
            


        if not x_vals or not y_vals:
            tk.messagebox.showwarning("No Data", "No valid data found for the selected variables.")
            return

        plt.figure(figsize=(8, 6))
        plt.scatter(x_vals, y_vals, alpha=0.7, edgecolors='k')
        plt.xlabel(x_var if x_var != "None" else "Index")
        plt.ylabel(y_var)
        plt.title(f"{y_var} vs {x_var if x_var != 'None' else 'Index'}")
        plt.grid(True)
        plt.show()


    def print_selected(self):
        selected_options = {
            key: [val for val, var in self.check_vars[key].items() if var.get()]
            for key in OPTION_KEYS
        }
        print("Selected options: ", selected_options)
        print("Selected folder: " + str(self.selected_folder)) 


    def select_folder(self):
        self.selected_folder = filedialog.askdirectory(initialdir=DEFAULT_PATH) + "/"
        if self.selected_folder:
            self.parse_cpp_files()
            self.create_checkboxes()
            for key, vars_dict in self.check_vars.items():
                for var_name, var in vars_dict.items():
                    var.set(True)  # Set each checkbox variable to True (checked)

    def parse_cpp_files(self):
        self.cpp_files = []
        self.option_values = {key: set() for key in OPTION_KEYS}

        for filename in os.listdir(self.selected_folder):
            if not filename.endswith(".cpp"):
                continue

            parts = filename[:-4].split("_")  # remove .cpp
            options = {}
            unmatched_parts = parts.copy()

            for key, valid_values in OPTION_KEYWORDS.items():
                for val in valid_values:
                    for part in unmatched_parts:
                        if val.lower() == part.lower():
                            options[key] = val
                            unmatched_parts.remove(part)
                            break
                    if key in options:
                        break

            if options:  # At least 1 match
                self.cpp_files.append((filename, options))
                for k, v in options.items():
                    self.option_values[k].add(v)

        if not self.cpp_files:
            messagebox.showwarning("No valid files", "No valid .cpp files found in the selected folder.")
        else:
            print(f"Parsed {len(self.cpp_files)} valid .cpp files.")

    def create_checkboxes(self):
        for widget in self.check_frame.winfo_children():
            widget.destroy()

        if not any(self.option_values.values()):
            messagebox.showinfo("No options", "No recognizable options found in filenames.")
            return

        for key in OPTION_KEYS:
            group = tk.LabelFrame(self.check_frame, text=key)
            group.pack(side=tk.LEFT, padx=5, pady=5)
            self.check_vars[key] = {}

            for val in OPTION_KEYWORDS[key]:
                if val in self.option_values[key]:
                    var = tk.BooleanVar()
                    cb = tk.Checkbutton(group, text=val, variable=var)
                    cb.pack(anchor="w")
                    self.check_vars[key][val] = var


    def load(self):
        self.loaded_data = []
        # Get selected options from all checkboxes
        selected = {
            key: [val for val, var in self.check_vars[key].items() if var.get()]
            for key in OPTION_KEYS
        }

        # Sanity check
        if not all(selected[k] for k in OPTION_KEYS):
            messagebox.showwarning("Incomplete Selection", "Please select at least one option in each category.")
            return

        # Get selected problem sets (assume only one for now)
        problem_sets = selected["Problem_Set"]
        if len(problem_sets) != 1:
            messagebox.showwarning("Invalid Selection", "Please select exactly one Problem Set.")
            return

        problem_set = problem_sets[0]

        # Remove Problem_Set from option keys for generating combinations
        combination_keys = [k for k in OPTION_KEYS if k != "Problem_Set"]
        all_combinations = list(product(*[selected[k] for k in combination_keys]))

        self.loaded_data = []

        for combination in all_combinations:
            combo_dict = dict(zip(combination_keys, combination))
            filename_base = f"{combo_dict['Hyper_Parameter_Scan']}_{combo_dict['Implementation']}_{combo_dict['Optimization_Type']}_{combo_dict['Non_linearity']}_{combo_dict['Update_type']}_{combo_dict['Chaotic_Amplitude_Control']}"

            if problem_set == "g05":
                for N in G05_N_VALUES:
                    for p in G05_PROBLEMS:
                        txt_name = f"{filename_base}_g05_{N}_{p}.txt"
                        if os.path.exists(os.path.join(self.selected_folder, txt_name)):
                            self.loaded_data.append(Data(self.selected_folder, txt_name))
            elif problem_set == "GsetUnweighted":
                for p in GSET_UNWEIGHTED:
                    txt_name = f"{filename_base}_G{p}.txt"
                    if os.path.exists(os.path.join(self.selected_folder, txt_name)):
                        self.loaded_data.append(Data(self.selected_folder, txt_name))
            elif problem_set == "GsetWeighted":
                for p in GSET_WEIGHTED:
                    txt_name = f"{filename_base}_G{p}.txt"
                    if os.path.exists(os.path.join(self.selected_folder, txt_name)):
                        self.loaded_data.append(Data(self.selected_folder, txt_name))

        if not self.loaded_data:
            messagebox.showwarning("No Data", "No matching data files found.")
        else:
            messagebox.showinfo("Loaded", f"Loaded {len(self.loaded_data)} data files.")
            # Example: You could now call self.plot_data(loaded_data) or store it  

        self.update_scatter_tab()

    def get_group_for_var(self, var_name):
        for group_name, var_list in groups:
            if var_name in var_list:
                return group_name
        return None  # Not found

    def parse_data_file(self, file_path):
        data = []
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()
                for i in range(0, len(lines) - 2, 2):
                    sample = {}
                    for line in lines[i:i+2]:
                        entries = line.strip().split(",")
                        for entry in entries:
                            if "=" in entry:
                                key, val = entry.split("=")
                                key = key.strip()
                                val = val.strip()
                                try:
                                    val = float(val)
                                except ValueError:
                                    pass
                                sample[key] = val
                    data.append(sample)
            return pd.DataFrame(data)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None



    def barplot(self):
        if self.barplot_problem_set_var.get() == "GsetWeighted":
            N_s = [800]*12 + [2000]*12
            problems = [6,7,8,9,10,11,12,13,18,19,20,21,27,28,29,30,31,32,33,34,39,40,41,42]
            num_rows = 2
        elif self.barplot_problem_set_var.get() == "GsetUnweighted":
            N_s = [800]*9 + [2000]*9 + [1000]*9
            problems = [1,2,3,4,5,14,15,16,17,22,23,24,25,26,35,36,37,38,43,44,45,46,47,51,52,53,54]
            num_rows = 3
        elif self.barplot_problem_set_var.get() == "g05":
            N_s = [60]*10 + [80]*10 + [100]*10
            problems = list(range(10)) * 3
            num_rows = 3
        else:
            print("Unknown benchmark set.")
            return
        print(self.barplot_problem_set_var.get())
        
        results = []
        best_known_values = []
        for N, problem in zip(N_s, problems):
            problem_name = None
            if self.barplot_problem_set_var.get() == "g05":
                problem_name = f"g05_{N}_{problem}"
            elif self.barplot_problem_set_var.get() == "GsetWeighted" or self.barplot_problem_set_var.get() == "GsetUnweighted":
                problem_name = f"G{problem}"
                print(problem_name)

            values_for_problem = []
            best_known_value_added = False
            for data_obj in self.loaded_data:
                if self.barplot_problem_set_var.get() != data_obj.options["Problem_Set"] or problem_name != data_obj.problem:
                    continue
                
                values_for_problem.append(data_obj.best_measures[self.barplot_measure_var.get()])
                if not best_known_value_added:
                    if self.barplot_measure_var.get() == "E_min":
                        best_known_values.append(data_obj.E_best_known if hasattr(data_obj, 'E_original_best_known') else None)
                        best_known_value_added = True
                    elif self.barplot_measure_var.get() == "E_min_original":
                        best_known_values.append(data_obj.E_original_best_known if hasattr(data_obj, 'E_original_best_known') else None)
                        best_known_value_added = True

            results.append(values_for_problem)
            

        print(f"Results: {results}")        
        print(f"Best known values: {best_known_values}")

        method_labels = []
        for data_obj in self.loaded_data:
            if self.barplot_problem_set_var.get() != data_obj.options["Problem_Set"]:
                continue
            label = (
                f"{data_obj.options['Optimization_Type']} {data_obj.options['Update_type']} {data_obj.options['Non_linearity']} "
                + ("Physical" if data_obj.options["Implementation"] == "Physical" else "Algorithm")
                + (" (BayesOpt)" if data_obj.options["Hyper_Parameter_Scan"] == "BayesianOptimization" else " (Grid)")
            )

            if label not in method_labels:
                method_labels.append(label)


        results = np.array(results)
        num_cols = int(np.ceil(len(N_s) / num_rows))
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(30, 15), constrained_layout=True)

        colors = sns.color_palette("colorblind")
        hatches_template = ['/', '\\', 'x', '|', '-', '+', 'o', 'O', '.', '*']
        hatches = [hatches_template[i % len(hatches_template)] for i in range(len(method_labels))]

        for idx, (N, problem) in enumerate(zip(N_s, problems)):
            row = idx // num_cols
            col = idx % num_cols
            ax = axs[row, col] if num_rows > 1 else axs[col]
            values = results[idx]
            finite_vals = values[np.isfinite(values)]
            min_val = min(finite_vals) if len(finite_vals) > 0 else 1
            max_val = max(finite_vals) if len(finite_vals) > 0 else 1
            if self.barplot_measure_var.get() in ["E_min", "E_min_original"]:
                max_val = best_known_values[idx] if best_known_values and best_known_values[idx] is not None else max_val

            if self.barplot_normalize.get():
                normalized = [v / max_val if np.isfinite(v) else float('inf') for v in values]
            else:
                normalized = [v if np.isfinite(v) else float('inf') for v in values]
            

            for i, (val, color, hatch) in enumerate(zip(normalized, colors, hatches)):
                height = val if val != float('inf') else 1
                ax.bar(i, height, color=color, edgecolor='black', hatch=hatch)
                if val == float('inf'):
                    label = '+∞'
                    y = 0.5 if not self.barplot_log_var.get() else 10 ** ((np.log10(min_val / max_val) + np.log10(1)) / 2)
                    ax.text(i, y, label, ha='center', va='center', fontsize=14, bbox=dict(facecolor='white', edgecolor='black'))

            if self.barplot_problem_set_var.get() == "g05":
                ax.set_title(f"g05_{N}_{problem}", fontsize=16)
            elif self.barplot_problem_set_var.get() == "GsetWeighted" or self.barplot_problem_set_var.get() == "GsetUnweighted":
                ax.set_title(f"G{problem}", fontsize=16)

            ax.set_xticks([])
            if self.barplot_log_var.get():
                ax.set_yscale('log')

            # Add red dashed line at best known value
            if self.barplot_normalize.get():
                if self.barplot_measure_var.get() in ["E_min", "E_min_original"]:
                    best_line = 1
                    ax.axhline(y=best_line, color='black', linestyle='--', linewidth=2, label='Best known',zorder=-10 )
            else:
                if self.barplot_measure_var.get() in ["E_min", "E_min_original"]:
                    best_line = best_known_values[idx] if best_known_values and best_known_values[idx] is not None else None
                    ax.axhline(y=best_line, color='black', linestyle='--', linewidth=2, label='Best known',zorder=-10 )
                

        legend_elements = [
            Patch(facecolor=colors[i % len(colors)], edgecolor='black', hatch=hatches[i], label=method_labels[i])
            for i in range(len(method_labels))
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=12, frameon=False)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(self.barplot_save_path_var.get() + self.barplot_name_var.get() + ".pdf")
        plt.show()





    def barplot_new(self):
        if self.barplot_problem_set_var.get() == "GsetWeighted":
            N_s = [800]*12 + [2000]*12
            problems = [6,7,8,9,10,11,12,13,18,19,20,21,27,28,29,30,31,32,33,34,39,40,41,42]
            num_rows = 2
        elif self.barplot_problem_set_var.get() == "GsetUnweighted":
            N_s = [800]*9 + [2000]*9 + [1000]*9
            problems = [1,2,3,4,5,14,15,16,17,22,23,24,25,26,35,36,37,38,43,44,45,46,47,51,52,53,54]
            num_rows = 3
        elif self.barplot_problem_set_var.get() == "g05":
            N_s = [60]*10 + [80]*10 + [100]*10
            problems = list(range(10)) * 3
            num_rows = 3
        else:
            print("Unknown benchmark set.")
            return

        results = []
        best_known_values = []
        for N, problem in zip(N_s, problems):
            if self.barplot_problem_set_var.get() == "g05":
                problem_name = f"g05_{N}_{problem}"
            else:
                problem_name = f"G{problem}"

            values_for_problem = []
            best_known_value_added = False
            for data_obj in self.loaded_data:
                if self.barplot_problem_set_var.get() != data_obj.options["Problem_Set"] or problem_name != data_obj.problem:
                    continue

                values_for_problem.append(data_obj.best_measures[self.barplot_measure_var.get()])
                if not best_known_value_added:
                    if self.barplot_measure_var.get() == "E_min":
                        best_known_values.append(data_obj.E_best_known if hasattr(data_obj, 'E_original_best_known') else None)
                        best_known_value_added = True
                    elif self.barplot_measure_var.get() == "E_min_original":
                        best_known_values.append(data_obj.E_original_best_known if hasattr(data_obj, 'E_original_best_known') else None)
                        best_known_value_added = True

            results.append(values_for_problem)

        # Mapping Optimization_Type → short legend label
        opt_map = {
            "GradientDescent": "GD-IM",
            "Momentum": "MOM-IM",
            "Adam": "ADAM-IM",
            "FirstOrderAdam": "1-ADAM-IM"
        }

        method_labels = []
        for data_obj in self.loaded_data:
            if self.barplot_problem_set_var.get() != data_obj.options["Problem_Set"]:
                continue
            label = (
                f"{opt_map.get(data_obj.options['Optimization_Type'],data_obj.options['Optimization_Type'])} {data_obj.options['Non_linearity']} "
                + ("Algorithmic" if data_obj.options["Implementation"] == "Algorithmic" else "")
            )
            if label not in method_labels:
                method_labels.append(label)

        results = np.array(results)
        num_methods = len(method_labels)
        num_cols = int(np.ceil(len(N_s) / num_rows))

        fig_width = max(16, len(N_s) * 0.5 / num_rows)
        fig_height = num_rows * 4
        fig, axs = plt.subplots(nrows=num_rows, ncols=1, figsize=(fig_width, fig_height), constrained_layout=True)
        #bg = (247/255, 245/255, 238/255)  # your color
        #bg = (224/255, 220/255, 203/255)
        bg = "white"
       
        fig.patch.set_facecolor(bg)   # figure background

        for ax in np.ravel(axs):      # iterate over all subplots
            ax.set_facecolor(bg)      # axes background



        if num_rows == 1:
            axs = [axs]

        # -------------------------------------------------
        # 🌿 Print‑friendly pastel palette (distinct in grayscale)
        # -------------------------------------------------
        
        #colors = [    (161/255, 34/255, 32/255),    (97/255, 114/255, 35/255),    (0/255, 100/255, 120/255),    (158/255, 115/255, 32/255),]

        #colors = [(227/255,  98/255, 101/255),   (215/255, 168/255, 204/255),   (142/255, 205/255, 211/255),   (219/255, 220/255, 112/255)]
        #colors = [(111/255, 161/255, 194/255), (123/255, 179/255, 159/255), (217/255, 137/255, 130/255), (210/255, 178/255,  76/255)]
        #colors = [(164/255,  79/255, 111/255), (116/255, 157/255, 175/255), (144/255, 176/255, 149/255), (205/255, 183/255, 109/255)]
        colors = [(176/255, 67/255, 87/255), (88/255, 132/255, 181/255), (86/255, 158/255, 119/255), (201/255, 164/255, 63/255)]
        
        # -------------------------------------------------

        hatches = [""] * num_methods
        bar_edgecolor = None

        bar_width = 1 / num_methods

        group_spacing = 0.3
        

        for row in range(num_rows):
            ax = axs[row]
            ax.tick_params(axis='y', which='major', labelsize=32, width=2.2, length=9)

            # -------------------------------------------------
            # NEW: subtle dashed horizontal gridlines
            # -------------------------------------------------
            
            ax.set_axisbelow(True)

            # Only use default grid if we are NOT in normalized E_min_original mode
            use_default_grid = not (self.barplot_normalize.get() and self.barplot_measure_var.get() == "E_min_original")

            if use_default_grid:
                ax.yaxis.grid(
                    True,
                    which="major",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.4,
                    color="0.25",
                    zorder=0
                )
            else:
                ax.yaxis.grid(False, which="major")  # we'll draw custom gridlines later
            # -------------------------------------------------

            start_idx = row * num_cols
            end_idx = min((row + 1) * num_cols, len(N_s))

            x_positions = []
            x_labels = []
            current_x = 0
            row_values = []
            

            if self.barplot_normalize.get() and self.barplot_measure_var.get() in ["E_min", "E_min_original"]:
                ax.axhline(y=1, color='black', linestyle='--',zorder=0, linewidth=2)

            for idx in range(start_idx, end_idx):
                values = results[idx]
                finite_vals = values[np.isfinite(values)]
                min_val = min(finite_vals) if len(finite_vals) > 0 else 1
                max_val = max(finite_vals) if len(finite_vals) > 0 else 1
                if self.barplot_measure_var.get() in ["E_min", "E_min_original"]:
                    max_val = best_known_values[idx] if best_known_values and best_known_values[idx] is not None else max_val
                
                # === NEW: annotate the baseline used for normalization above each problem group ===

                if self.barplot_normalize.get():
                    if self.barplot_measure_var.get() in ["E_min", "E_min_original"]:
                        normalized = [v / max_val if np.isfinite(v) else float('inf') for v in values]
                    else:
                        normalized = [v / min_val if np.isfinite(v) else float('inf') for v in values]
                else:
                    normalized = [v if np.isfinite(v) else float('inf') for v in values]
                if self.barplot_normalize.get() and self.barplot_measure_var.get() in ["TTS_CPU", "TTS_Euler"]:
                    # Only annotate if we actually had finite values (otherwise min_val is a fallback)
                    if len(finite_vals) > 0:
                        base = min_val
                        # --- Auto-scale TTS_CPU ---
                        if self.barplot_measure_var.get() == "TTS_CPU":
                            # scale to s / ms / µs / ns
                            if base >= 1:
                                show = base
                                unit = "s"
                            elif base >= 1e-3:
                                show = base * 1e3     # ms
                                unit = "ms"
                            elif base >= 1e-6:
                                show = base * 1e6     # µs
                                unit = "µs"
                            else:
                                show = base * 1e9     # ns
                                unit = "ns"
                        else:
                            show = base
                            unit = ""

                        # --- Avoid values that round to 0.xx in current unit ---
                        # If it would print as 0.00 or 0.0, drop to a smaller unit
                        if unit == "ms" and show < 0.01:
                            show *= 1000
                            unit = "µs"

                        if unit == "µs" and show < 0.01:
                            show *= 1000
                            unit = "ns"

                        # --- k-format (AFTER scaling) ---
                        suffix = ""
                        if show >= 10000:
                            show /= 1000.0
                            suffix = "k"

                        abs_show = abs(show)

                        # --- Your original fixed-point formatting rules ---
                        if abs_show >= 100:
                            base_str = f"{show:.0f}{unit}"
                        elif abs_show >= 10:
                            base_str = f"{show:.1f}{unit}"
                        else:
                            base_str = f"{show:.2f}{unit}"

                        # --- Remove trailing ".0" (e.g., "16.0 s" → "16 s") ---
                        if base_str.endswith(".0" + unit):
                            base_str = base_str.replace(".0" + unit, unit)

                        # --- Add k suffix ---
                        base_str = base_str + suffix


                        # Determine x position: center of the group you're plotting now
                        x_center = current_x + bar_width * ((num_methods - 1) / 2)

                        # Determine y position: a bit above the tallest finite normalized bar in this group
                        finite_norm = [v for v in normalized if np.isfinite(v)]
                        fontsize_numbers = 24
                        #if we plot TTS_CPU change fontsize_numbers
                        if self.barplot_measure_var.get() == "TTS_CPU":
                            fontsize_numbers = 24
                        else:
                            fontsize_numbers = 30
                        if finite_norm:
                            group_max = max(finite_norm)
                            # Slightly higher multiplier on log-scale for more headroom
                            y_text = group_max * (1.10 if not self.barplot_log_var.get() else 1.01)

                            ax.text(
                                x_center, y_text, base_str,
                                ha='center', va='bottom',
                                fontsize=fontsize_numbers, color='black'
                            )
                # === END NEW ===

                row_values.extend([v for v in normalized if v != float('inf')])
                if row_values:
                    ylimit = max(row_values) * 1.1
                    if self.barplot_log_var.get():
                        ylimit = int(np.ceil(np.log10(ylimit)))*1.5
                for i, (val, color, hatch) in enumerate(zip(normalized, colors, hatches)):
                    x = current_x + i * bar_width
                    #max float
                    height = val if val != float('inf') else 0
                    ax.bar(
                        x,
                        height,
                        width=bar_width,
                        color=color,
                        edgecolor=bar_edgecolor,
                        hatch=hatch
                    )
                    """ font_size = 19
                    if self.barplot_problem_set_var.get() == "GsetWeighted":
                        font_size = 15

                     if val == float('inf'):
                        # Place a vertical "+∞" label on the bar
                        ax.text(
                        x,
                        pow(10, (ylimit - 1) / 2),
                        '+∞',
                        ha='center',
                        va='center',
                        fontsize=font_size,
                        rotation=90,
                        rotation_mode='anchor',   # <-- IMPORTANT: keeps box centered
                        bbox=dict(
                            facecolor='white',
                            edgecolor='black',
                            boxstyle='round,pad=0.02'
                        )
                    ) """

                problem_label = (
                    f"{N_s[idx]}.{problems[idx]}"
                    if self.barplot_problem_set_var.get() == "g05"
                    else f"G{problems[idx]}"
                )
                x_labels.append(problem_label)
                #if num_methods is even, center the group between two bars:
                x_positions.append(current_x + bar_width*((num_methods-1)/2))  #center of the group

                current_x += num_methods * bar_width + group_spacing

            ax.set_xticks(x_positions)
            x_label_fontsize = 32
            ax.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=x_label_fontsize)


            if self.barplot_log_var.get():
                ax.set_yscale('log')
                # Make minor log ticks more visible
                ax.minorticks_off()
                if row_values:
                    ymin = max(min(row_values) * 0.85, 1e-2)
                    ymax = max(row_values) * 1.1
                min_exp = int(np.floor(np.log10(ymin))) if ymin > 0 else -1
                max_exp = int(np.ceil(np.log10(ymax)))
                ax.set_yticks([10**i for i in range(min_exp, max_exp+1)])
                ax.get_yaxis().set_major_formatter(FuncFormatter(lambda y, _: f"$10^{{{int(np.log10(y))}}}$"))
                ymin = min(row_values) * 0.85 if row_values else 1e-2
                if row_values:
                    max_val = max(row_values)
                    n = np.ceil(np.log10(max_val))
                    if self.barplot_problem_set_var.get() == "g05":
                        factor = 2
                    else:
                        factor = 1.5
                    ymax = factor * 10**n
                else:
                    ymax = 1
            else:
                ymin = 0
                ymax = 1.1 * max(row_values) if row_values else 1.1

            if self.barplot_measure_var.get() in ["E_min", "E_min_original"]:
                ymin = min(row_values) if row_values else 0
                ymax = max(max(row_values),1) if row_values else 1
                ymin -= abs(ymin - ymax)*0.05
                ymax += abs(ymin - ymax)*0.05

            if self.barplot_log_var.get():
                ymin = min(ymin, 10**(-0.25))
            ax.set_ylim(ymin, ymax)


            if self.barplot_measure_var.get() == "E_min_original" and self.barplot_normalize.get():
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y*100:.1f}%"))

            # ----------------------------
            # Dynamic LaTeX y-axis label
            # ----------------------------
            measure = self.barplot_measure_var.get()

            # Map to LaTeX versions
            latex_map = {
                "TTS_Euler": r"\mathrm{TTS}",
                "TTS_CPU": r"\mathrm{TTS_{CPU}}",
                "SR_tr": r"\mathrm{SR_{tr}}",
                "SR": r"\mathrm{SR}",
                "E_min": r"\mathrm{E_{min}}",
                "E_min_original": r"\mathrm{E_{min}^{original}}",
            }

            y_label = latex_map.get(measure, measure)

            #if self.barplot_measure_var.get() == "TTS_Euler" and self.barplot_problem_set_var.get() != "g05":
            #    y_label = r"\mathrm{TTT_{99.5\%}}"
            #elif self.barplot_measure_var.get() == "TTS_CPU" and self.barplot_problem_set_var.get() != "g05":
            #    y_label = r"\mathrm{TTT_{CPU,99.5\%}}"
            if self.barplot_measure_var.get() == "TTS_Euler":
                y_label = r"\mathrm{TTT}"
            elif self.barplot_measure_var.get() == "TTS_CPU":
                y_label = r"\mathrm{TTT_{CPU}}"
            elif self.barplot_measure_var.get() == "TTS_Euler_eta_corrected":
                y_label = r"\mathrm{TTT}\ \mathrm{rescaled}"
            
            # Add normalization prefix
            if self.barplot_normalize.get():
                y_label = rf"\mathrm{{Normalized}}\ {y_label}"

            # Apply LaTeX formatting
            y_label = f"${y_label}$"
            
            y_label2 = self.barplot_measure_var.get()
            if y_label2 == "E_min_original" and self.barplot_normalize.get():
                if self.barplot_problem_set_var.get() == "g05":
                    y_label = r"% of max cut value"
                else:
                    # Applies to GsetWeighted or GsetUnweighted
                    y_label = r"% of best known cut value"

            if not self.barplot_log_var.get():
                # Clean percentage ticks for SR_tr and SR
                if self.barplot_measure_var.get() in ["SR_tr", "SR"]:
                    ymin, ymax = ax.get_ylim()

                    target_ticks = 7

                    # Steps in normalized units:
                    # 0.05 = 5%, 0.10 = 10%, etc.
                    allowed_steps = np.array([0.05, 0.10, 0.20, 0.25, 0.50, 1.00])

                    yrange = ymax - ymin
                    ideal_step = yrange / target_ticks

                    # Choose the smallest clean step that is not too dense
                    possible_steps = allowed_steps[allowed_steps >= ideal_step]
                    if len(possible_steps) > 0:
                        step = possible_steps[0]
                    else:
                        step = allowed_steps[-1]

                    tick_start = np.ceil(ymin / step) * step
                    tick_end = np.floor(ymax / step) * step

                    ticks = np.arange(tick_start, tick_end + step / 2, step)

                    # Safety fallback if range is very small
                    if len(ticks) == 0:
                        ticks = np.array([round((ymin + ymax) / 2, 2)])

                    ax.set_yticks(ticks)

                    def clean_percent_label(y, pos):
                        p = y * 100
                        return f"{int(round(p))}%"

                    ax.yaxis.set_major_formatter(FuncFormatter(clean_percent_label))


                # Keep your detailed tick spacing for normalized E_min_original
                elif self.barplot_normalize.get() and self.barplot_measure_var.get() == "E_min_original":
                    ymin, ymax = ax.get_ylim()
                    target_ticks = 8

                    yrange = ymax - ymin
                    ideal_step = yrange / target_ticks

                    snapped_step = np.ceil(ideal_step / 0.001) * 0.001

                    m = int(np.floor((1.0 - ymin) / snapped_step))

                    ticks = np.linspace(1.0, 1.0 - m * snapped_step, m + 1)

                    ax.set_yticks(ticks)

                    # Custom gridlines: draw all major gridlines except at 100%
                    grid_kwargs = dict(
                        color="0.25",
                        linestyle="--",
                        linewidth=2.0,
                        alpha=0.85,
                        zorder=-1
                    )

                    for y in ax.get_yticks():
                        if abs(y - 1.0) > 1e-12:
                            ax.axhline(y, **grid_kwargs)

                    def clean_percent_label(y, pos):
                        p = y * 100
                        if abs(p - 100) < 1e-8:
                            return "100%"
                        s = f"{p:.1f}%".replace(",", ".")
                        if s.endswith(".0%"):
                            s = s.replace(".0%", "%")
                        return s

                    ax.yaxis.set_major_formatter(FuncFormatter(clean_percent_label))

        legend_elements = [
            Patch(facecolor=colors[i % len(colors)], edgecolor=None, hatch="", label=method_labels[i])
            for i in range(num_methods)
        ]

        fig.text(-0.0435, 0.5, y_label, va='center', rotation='vertical', fontsize=40)

        if self.barplot_problem_set_var.get() == "GsetWeighted":
            legend_y_pos = 1.175
        else:
            legend_y_pos = 1.115
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, legend_y_pos),
                ncol=2, fontsize=32, frameon=False, borderaxespad=0.2)
        pdf_path = self.barplot_save_path_var.get() + self.barplot_name_var.get() + ".pdf"
        plt.savefig(pdf_path,
                    bbox_inches="tight")
        

        if sys.platform.startswith("darwin"):        # macOS
            subprocess.Popen(["open", pdf_path])
        elif sys.platform.startswith("win"):         # Windows
            os.startfile(pdf_path)
        else:                                        # Linux (xdg-open)
            subprocess.Popen(["xdg-open", pdf_path])


    def export_gset_latex(self):
        # Require data
        if not getattr(self, "loaded_data", None):
            messagebox.showwarning("No Data", "Load data files first.")
            return

        # Selected options from checkboxes (empty list = no filter on that key)
        selected = {
            key: [val for val, var in self.check_vars.get(key, {}).items() if var.get()]
            for key in OPTION_KEYS
        }

        METHOD_KEY = "Optimization_Type"

        def matches_selected_except_method(d):
            for k, vals in selected.items():
                if k == METHOD_KEY:
                    continue
                if vals and d.options.get(k) not in vals:
                    return False
            return True

        METHOD_MAP = {
            "gradientdescent": "GD-IM",
            "momentum": "MOM-IM",
            "adam": "ADAM-IM",
            "firstorderadam": "1-ADAM-IM",
        }

        def detect_method_tag(d):
            v = d.options.get(METHOD_KEY)
            if isinstance(v, str):
                return METHOD_MAP.get(v.replace(" ", "").lower())
            return None

        results = {}  # { "G5": {"GD-IM": best, "MOM-IM": best, "ADAM-IM": best, "1-ADAM-IM": best, "best_known": int} }
        for d in self.loaded_data:
            if not isinstance(getattr(d, "problem", None), str) or not d.problem.startswith("G"):
                continue
            if not matches_selected_except_method(d):
                continue

            tag = detect_method_tag(d)
            if tag not in METHOD_MAP.values():
                continue

            best_found = d.best_measures.get("E_min_original", float("-inf"))
            key = d.problem
            if key not in results:
                results[key] = {
                    "GD-IM": None, "MOM-IM": None, "ADAM-IM": None, "1-ADAM-IM": None,
                    "best_known": getattr(d, "E_original_best_known", None)
                }

            cur = results[key][tag]
            if (cur is None) or (best_found > cur):
                results[key][tag] = best_found
            if results[key]["best_known"] is None:
                results[key]["best_known"] = getattr(d, "E_original_best_known", None)

        if not results:
            messagebox.showinfo("No Gset matches", "No Gset problems match the selected options.")
            return

        def gnum(s):
            try:
                return int(s[1:])
            except Exception:
                return 10**9

        def to_int_or_none(v):
            if v is None or (isinstance(v, float) and v == float('-inf')):
                return None
            return int(round(v))

        rows = []
        rows.append("\\begin{table}[ht]")
        rows.append("\\centering")
        rows.append("\\begin{tabular}{lrrrrr}")
        rows.append("\\toprule")
        rows.append("Graph & Best known & 1-ADAM-IM & ADAM-IM & MOM-IM & GD-IM &  \\\\")
        rows.append("\\midrule")

        for g in sorted(results.keys(), key=gnum):
            r = results[g]
            gim = to_int_or_none(r["GD-IM"])
            mim = to_int_or_none(r["MOM-IM"])
            aim = to_int_or_none(r["ADAM-IM"])
            aim1 = to_int_or_none(r["1-ADAM-IM"])
            bk  = to_int_or_none(r["best_known"])

            # Compute the max across all five columns (ignoring missing)
            candidates = [v for v in (bk,aim1,aim,mim,gim) if v is not None]
            max_val = max(candidates) if candidates else None

            def maybe_bold(v):
                if v is None:
                    return "-"
                return f"\\textbf{{{v}}}" if (max_val is not None and v == max_val) else str(v)

            row = (
                f"{g} & "
                f"{maybe_bold(bk)} & "
                f"{maybe_bold(aim1)} & "
                f"{maybe_bold(aim)} & "
                f"{maybe_bold(mim)} & "
                f"{maybe_bold(gim)} \\\\"
            )
            rows.append(row)

        rows.append("\\bottomrule")
        rows.append("\\end{tabular}")
        rows.append("\\caption{Max-Cut $E_{\\min}^{\\text{original}}$ on \\textsc{Gset}: methods (1-ADAM-IM, ADAM-IM, MOM-IM, GD-IM) vs. best known. Bold indicates the highest value(s) per row.}")
        rows.append("\\label{tab:gset_methods_selected}")
        rows.append("\\end{table}")

        latex_table = "\n".join(rows)

        # Copy to clipboard and optionally save
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(latex_table)
        except Exception:
            pass

        save_path = filedialog.asksaveasfilename(
            title="Save LaTeX table",
            defaultextension=".tex",
            initialfile="gset_methods_table.tex",
            filetypes=[("LaTeX file", "*.tex"), ("All files", "*.*")]
        )
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(latex_table)
            messagebox.showinfo("Saved", f"LaTeX table saved to:\n{save_path}")
        else:
            messagebox.showinfo("Copied", "LaTeX table copied to clipboard.")


if __name__ == "__main__":
    root = tk.Tk()
    app = SimulationDataGUI(root)
    root.mainloop()
