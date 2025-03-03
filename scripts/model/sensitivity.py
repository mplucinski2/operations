from gurobipy import Model, GRB, quicksum
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from datetime import datetime
import os

# Set the seed for reproducibility
random.seed(42)

# Function to generate random travel times
def generate_random_travel_times(num_nodes, max_time=20):
    DT = {}
    for i in range(num_nodes):
        for j in range(i, num_nodes):  # Only go one way to avoid overwriting
            if i == j:
                DT[(i, j)] = 0
            else:
                time = random.randint(1, max_time)
                DT[(i, j)] = time
                DT[(j, i)] = time  # Make sure it's symmetric
    return DT

# Function to generate a dict with specified values
def generate_random_dict(num_entries, value_range):
    return {i: random.randint(value_range[0], value_range[1]) for i in range(1, num_entries + 1)}

# Function to generate a list with a specified range
def generate_list(start, end):
    return list(range(start, end + 1))

# Function to plot sensitivity analysis results
def plot_sensitivity(parameter_values, objective_values, title, xlabel, ylabel="Objective Value", 
                    save_path=None, show_plot=True, include_baseline=None):
    plt.figure(figsize=(10, 6))
    
    # Plot the results
    plt.plot(parameter_values, objective_values, 'o-', color="royalblue", linewidth=2)
    
    # If baseline value is provided, add a vertical line
    if include_baseline is not None:
        baseline_idx = parameter_values.index(include_baseline)
        baseline_obj = objective_values[baseline_idx]
        plt.axvline(x=include_baseline, color='red', linestyle='--', alpha=0.7)
        plt.annotate(f'Baseline: {include_baseline}', 
                    xy=(include_baseline, min(objective_values)),
                    xytext=(0, -30), textcoords='offset points',
                    ha='center', color='red')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    
    # Add value labels to each point
    for i, (x, y) in enumerate(zip(parameter_values, objective_values)):
        plt.annotate(f'{y:.0f}', 
                    xy=(x, y),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center')
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

# Main function for sensitivity analysis
def run_sensitivity_analysis(parameter_name, parameter_range, base_settings, time_limit=300, output_dir="results"):
    """
    Perform sensitivity analysis on a specified parameter.
    
    Args:
        parameter_name (str): Name of the parameter to vary
        parameter_range (list): Values of the parameter to test
        base_settings (dict): Base values for all parameters
        time_limit (int): Time limit in seconds for each optimization
        output_dir (str): Directory to save results
    
    Returns:
        dict: Results of the sensitivity analysis
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize results storage
    results = {
        'parameter_values': parameter_range,
        'objective_values': [],
        'accepted_orders': [],
        'rejected_orders': [],
        'solve_times': [],
        'gaps': []
    }
    
    # Run optimization for each parameter value
    for param_value in parameter_range:
        print(f"\n{'-'*50}")
        print(f"Running with {parameter_name} = {param_value}")
        print(f"{'-'*50}")
        
        # Update parameter in settings
        current_settings = base_settings.copy()
        
        # Special handling for different parameter types
        if parameter_name == "CA":  # Capacity
            current_settings["CA"] = param_value
        elif parameter_name == "alpha":  # Earliness/tardiness penalty
            current_settings["alpha"] = param_value
        elif parameter_name == "num_batches":
            current_settings["B"] = generate_list(1, param_value)
            current_settings["B0"] = [0] + current_settings["B"]
        elif parameter_name == "num_vehicles":
            current_settings["V"] = generate_list(1, param_value)
        elif parameter_name == "temperature_range":
            # Adjust temperature bounds symmetrically around center point
            t_lb_original = current_settings["t_lb"].copy()
            t_ub_original = current_settings["t_ub"].copy()
            
            for i in current_settings["I"]:
                center = (t_lb_original[i] + t_ub_original[i]) / 2
                half_range = param_value / 2
                current_settings["t_lb"][i] = max(0, center - half_range)
                current_settings["t_ub"][i] = center + half_range
        elif parameter_name == "shelf_life_factor":
            # Adjust shelf life by multiplying the original
            for i in current_settings["I"]:
                current_settings["S"][i] = int(current_settings["S"][i] * param_value)
        elif parameter_name == "revenue_factor":
            # Adjust revenue by multiplying the original
            for i in current_settings["I"]:
                current_settings["R"][i] = int(current_settings["R"][i] * param_value)
        
        # Run the model with current parameter settings
        model_result = run_model(current_settings, time_limit)
        
        # Store results
        results['objective_values'].append(model_result['objective'])
        results['accepted_orders'].append(model_result['accepted'])
        results['rejected_orders'].append(model_result['rejected'])
        results['solve_times'].append(model_result['solve_time'])
        results['gaps'].append(model_result['gap'])
    
    # Generate timestamp for saving files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        parameter_name: results['parameter_values'],
        'objective': results['objective_values'],
        'accepted_orders': results['accepted_orders'],
        'rejected_orders': results['rejected_orders'],
        'solve_time': results['solve_times'],
        'gap': results['gaps']
    })
    csv_path = f"{output_dir}/sensitivity_{parameter_name}_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    
    # Plot results
    plot_path = f"{output_dir}/sensitivity_{parameter_name}_{timestamp}.png"
    plot_sensitivity(
        results['parameter_values'], 
        results['objective_values'],
        f"Sensitivity Analysis: Objective vs {parameter_name}",
        parameter_name,
        save_path=plot_path,
        include_baseline=base_settings.get(parameter_name)
    )
    
    # Plot secondary metrics
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(results['parameter_values'], results['accepted_orders'], 'o-', color="green")
    plt.plot(results['parameter_values'], results['rejected_orders'], 'o-', color="red")
    plt.title(f"Orders Accepted/Rejected vs {parameter_name}")
    plt.xlabel(parameter_name)
    plt.ylabel("Number of Orders")
    plt.grid(True, alpha=0.3)
    plt.legend(["Accepted", "Rejected"])
    
    plt.subplot(1, 2, 2)
    plt.plot(results['parameter_values'], results['solve_times'], 'o-', color="purple")
    plt.title(f"Solution Time vs {parameter_name}")
    plt.xlabel(parameter_name)
    plt.ylabel("Time (seconds)")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sensitivity_{parameter_name}_metrics_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nResults saved to {csv_path} and plots saved to {output_dir}")
    return results

# Main optimization model function
def run_model(settings, time_limit=300):
    """
    Run the pharmaceutical delivery optimization model with specified settings.
    
    Args:
        settings (dict): Dictionary containing all model parameters
        time_limit (int): Time limit in seconds
    
    Returns:
        dict: Results including objective value and solution metrics
    """
    # Extract parameters from settings
    I = settings.get("I", generate_list(1, 20))  # Default to 20 orders
    B = settings.get("B", generate_list(1, 5))   # Default to 5 batches
    B0 = settings.get("B0", [0] + B)
    V = settings.get("V", generate_list(1, 8))   # Default to 8 vehicles
    N = [0] + I  # Nodes: 0 = depot, others = orders
    
    # Parameters
    R = settings.get("R", generate_random_dict(len(I), (1000, 2000)))           # Revenue
    W = settings.get("W", generate_random_dict(len(I), (5, 25)))                # Weight
    t_lb = settings.get("t_lb", generate_random_dict(len(I), (1, 5)))           # Lower temp bounds
    t_ub = settings.get("t_ub", generate_random_dict(len(I), (6, 15)))          # Upper temp bounds
    CA = settings.get("CA", 25)                                                 # Capacity
    ST_LB = settings.get("ST_LB", generate_random_dict(len(I), (6, 10)))       # Feasibility checks
    ST_UB = settings.get("ST_UB", generate_random_dict(len(I), (15, 20)))      # Feasibility checks
    D = settings.get("D", generate_random_dict(len(I), (5, 15)))               # Due dates
    S = settings.get("S", generate_random_dict(len(I), (15, 25)))              # Shelf lives
    alpha = settings.get("alpha", 1)                                           # Coefficient
    M = settings.get("M", 1e7)                                                 # Cost weight
    
    # Travel time matrix
    DT = settings.get("DT", generate_random_travel_times(len(N), max_time=20))
    
    # Create and solve the model
    model = Model("PharmaceuticalDelivery")
    
    # Set time limit
    model.setParam(GRB.Param.TimeLimit, time_limit)
    
    # Decision variables (as in main_attempt1.py)
    a = model.addVars(I, vtype=GRB.BINARY, name="a")     # Order acceptance
    x = model.addVars(I, B, vtype=GRB.BINARY, name="x")  # Assign order i to batch b
    y = model.addVars(B, V, vtype=GRB.BINARY, name="y")  # Assign batch b to vehicle v
    z = model.addVars(I, vtype=GRB.BINARY, name="z")     # Disposal indicator
    t_b = model.addVars(B, vtype=GRB.CONTINUOUS, lb=0, name="t_b") # Batch temperature
    r_b = model.addVars(N, N, B, vtype=GRB.BINARY, name="r_b") # Routing within batch
    u_i = model.addVars(N, B, vtype=GRB.INTEGER, name="u_i")    # Subtour elim in routing
    s = model.addVars(B0, B0, V, vtype=GRB.BINARY, name="s")      # Scheduling of batches
    s_b_var = model.addVars(B, vtype=GRB.CONTINUOUS, name="s_b") # Start time of batch
    c_b_var = model.addVars(B, vtype=GRB.CONTINUOUS, name="c_b") # Completion time of batch
    u_bv = model.addVars(B0, V, vtype=GRB.INTEGER, name="u_bv")   # Subtour elim in scheduling
    E = model.addVars(I, vtype=GRB.CONTINUOUS, name="E")         # Earliness
    T = model.addVars(I, vtype=GRB.CONTINUOUS, name="T")         # Tardiness
    c = model.addVars(I, vtype=GRB.CONTINUOUS, name="c")         # Completion time of orders
    
    # Objective function
    DeliveryRevenue = quicksum(R[i]*a[i] for i in I)
    EarlinessTardinessCost = quicksum(alpha*R[i]*(E[i] + T[i]) for i in I)
    DisposalCost = quicksum(R[i]*z[i] for i in I)
    model.setObjective(DeliveryRevenue - (EarlinessTardinessCost + DisposalCost), GRB.MAXIMIZE)

    # Add all constraints exactly as in main_attempt1.py
    # Order assignment (9)
    model.addConstrs((quicksum(x[i,b] for b in B) == a[i] for i in I), name="AssignOrderToBatch")

    # Capacity (10)
    model.addConstrs((quicksum(x[i,b]*W[i] for i in I) <= CA for b in B), name="Capacity")

    # Temperature feasibility (11)
    model.addConstrs((t_lb[i] - M*(1 - x[i,b]) <= t_b[b] for i in I for b in B), name="TempLower")
    model.addConstrs((t_b[b] <= t_ub[i] + M*(1 - x[i,b]) for i in I for b in B), name="TempUpper")

    # Batch to vehicle assignment (12 - 13)
    model.addConstrs((quicksum(y[b,v] for v in V) <= quicksum(x[i,b] for i in I) for b in B), name="NonemptyBatchIfAssigned")
    model.addConstrs((x[i,b] <= quicksum(y[b,v] for v in V) for i in I for b in B), name="LinkBatchVehicle")
    model.addConstrs((quicksum(y[b,v] for v in V) <= 1 for b in B), name="OneVehiclePerBatch")

    # Routing constraints (14 - 15)
    model.addConstrs((r_b[0,i,b] + quicksum(r_b[j,i,b] for j in I if j!=i) == x[i,b] for b in B for i in I), name="RoutingInflow")
    model.addConstrs((r_b[i,0,b] + quicksum(r_b[i,j,b] for j in I if j!=i) == x[i,b] for b in B for i in I), name="RoutingOutflow")

    # Nonempty batch: route must start and end at depot (16 - 17)
    model.addConstrs((quicksum(x[i,b] for i in I) <= M * quicksum(r_b[0,j,b] for j in I) for b in B), name="StartFromDepot")
    model.addConstrs((quicksum(x[i,b] for i in I) <= M * quicksum(r_b[j,0,b] for j in I) for b in B), name="ReturnToDepot")

    # No loops (18)
    model.addConstrs((r_b[i,i,b] == 0 for b in B for i in N), name="NoSelfLoop")

    # Subtour elimination in routing (19 - 22)
    model.addConstrs((u_i[0,b] == 0 for b in B), name="DepotOrderPosition")
    model.addConstrs((u_i[i,b] + r_b[i,j,b] <= u_i[j,b] + M*(1 - r_b[i,j,b])
                      for b in B for i in N for j in I if i != j), name="RoutingSubtour1")
    model.addConstrs((u_i[i,b] <= M * quicksum(r_b[i,j,b] for j in I if j !=i) for b in B for i in I), name="RoutingSubtour2")
    model.addConstrs((u_i[i,b] <= quicksum(x[k,b] for k in I) for b in B for i in I), name="RoutingSubtour3")

    # Scheduling constraints (batches on vehicles) (23 - 24)
    model.addConstrs((s[0,i,v] + quicksum(s[j,i,v] for j in B if j !=i) == y[i,v] for v in V for i in B), name="SchedInflow")
    model.addConstrs((s[i,0,v] + quicksum(s[i,j,v] for j in B if j !=i) == y[i,v] for v in V for i in B), name="SchedOutflow")

    # Nonempty vehicle schedule start/end (25 - 26)
    model.addConstrs((quicksum(y[i,v] for i in B) <= M * quicksum(s[0,j,v] for j in B) for v in V), name="SchedStart")
    model.addConstrs((quicksum(y[i,v] for i in B) <= M * quicksum(s[j,0,v] for j in B) for v in V), name="SchedEnd")

    # No self loops in scheduling (27)
    model.addConstrs((s[b,b,v] == 0 for b in B0 for v in V), name="NoSelfLoopScheduling")

    # Subtour elimination in scheduling (28 - 31)
    model.addConstrs((u_bv[0, v] == 0 for v in V), name="SchedulingDepot")
    model.addConstrs((u_bv[i,v] + s[i,j,v] <= u_bv[j,v] + M*(1 - s[i,j,v]) for v in V for i in B for j in B if i != j), name="SchedulingSubtour1")
    model.addConstrs((u_bv[i,v] <= M * quicksum(s[i,j,v] for j in B if j != i) for v in V for i in B), name="SchedulingSubtour2")
    model.addConstrs((u_bv[i, v] <= quicksum(y[j, v] for j in B) for v in V for i in B), name="SchedulingSubtour3")

    # Batch timing constraints (32 - 35)
    model.addConstrs((c_b_var[i] <= s_b_var[j] + M*(1 - s[i,j,v]) for v in V for i in B for j in B if i !=j), name="BatchTiming1")
    model.addConstrs((s_b_var[j] <= c_b_var[i] + M*(1 - s[i,j,v]) for v in V for i in B for j in B if i !=j), name="BatchTiming2")
    model.addConstrs((s_b_var[i] <= M * (1 - s[0,i,v]) for v in V for i in B), name="BatchStartFrom0")
    model.addConstrs((0 <= s_b_var[i] + M * (1 - s[0,i,v]) for v in V for i in B), name="BatchStartNonNeg")

    # Completion times of orders (36 - 37)
    model.addConstrs((c[j] >= c[i] + DT[i,j] - M * (1 - r_b[i,j,b]) for b in B for i in I for j in I if i != j), name="OrderTimeForward")
    model.addConstrs((c[j] <= c[i] + DT[i,j] + M * (1 - r_b[i,j,b]) for b in B for i in I for j in I if i != j), name="OrderTimeBackward")

    # First visited order in batch (38 - 39)
    model.addConstrs((c[i] >= s_b_var[b] + DT[0,i] - M * (1 - r_b[0,i,b]) for b in B for i in I), name="FirstOrderTime1")
    model.addConstrs((c[i] <= s_b_var[b] + DT[0,i] + M * (1 - r_b[0,i,b]) for b in B for i in I), name="FirstOrderTime2")

    # Last visited order in batch (40 - 41)
    model.addConstrs((c_b_var[b] >= c[i] + DT[i,0] - M * (1 - r_b[i,0,b]) for b in B for i in I), name="LastOrderTime1")
    model.addConstrs((c[i] >= c_b_var[b] - M * (1 - r_b[i,0,b]) - DT[i,0] for b in B for i in I), name="LastOrderTime2")

    # Earliness and Tardiness (3 - 4)
    model.addConstrs((E[i] >= D[i] - c[i] - M * (1 - a[i]) for i in I), name="EarlinessDef")
    model.addConstrs((T[i] >= c[i] - D[i] - M * (1 - a[i]) for i in I), name="TardinessDef")

    # Disposal (6)
    model.addConstrs((c[i] - S[i] <= M * z[i] for i in I), name="DisposalDef")

    # Link disposal to order rejection
    model.addConstrs((z[i] >= 1 - a[i] for i in I), name="LinkDisposalLower")
    model.addConstrs((z[i] <= 1 - a[i] for i in I), name="LinkDisposalUpper")
    
    # Disable presolve
    model.setParam(GRB.Param.Presolve, 0)
    
    # Start timing
    start_time = time.time()
    
    # Optimize
    model.optimize()
    
    # End timing
    solve_time = time.time() - start_time
    
    # Collect results
    results = {
        'objective': float('nan'),
        'accepted': 0,
        'rejected': 0,
        'solve_time': solve_time,
        'gap': float('inf')
    }
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
            # Get objective value
            results['objective'] = model.objVal
            results['gap'] = model.MIPGap
            
            # Count accepted and rejected orders
            for i in I:
                if a[i].X > 0.5:
                    results['accepted'] += 1
                else:
                    results['rejected'] += 1
    
    return results

# Function to create baseline settings
def create_baseline_settings(num_orders=20):
    """
    Create baseline settings for the model.
    
    Returns:
        dict: Baseline parameter settings
    """
    # Create base parameter sets
    I = generate_list(1, num_orders)
    B = generate_list(1, 5)
    B0 = [0] + B
    V = generate_list(1, 8)
    N = [0] + I
    
    # Generate random parameters
    R = generate_random_dict(len(I), (1000, 2000)) 
    W = generate_random_dict(len(I), (5, 25))
    t_lb = generate_random_dict(len(I), (1, 5))
    t_ub = generate_random_dict(len(I), (6, 15))
    ST_LB = generate_random_dict(len(I), (6, 10))
    ST_UB = generate_random_dict(len(I), (15, 20))
    D = generate_random_dict(len(I), (5, 15))
    S = generate_random_dict(len(I), (15, 25))
    DT = generate_random_travel_times(len(N), max_time=20)
    
    # Baseline settings
    return {
        "I": I,
        "B": B,
        "B0": B0,
        "V": V,
        "N": N,
        "R": R,
        "W": W,
        "t_lb": t_lb,
        "t_ub": t_ub,
        "CA": 25,
        "ST_LB": ST_LB,
        "ST_UB": ST_UB,
        "D": D,
        "S": S,
        "alpha": 1,
        "M": 1e7,
        "DT": DT
    }

# Function to run a full sensitivity analysis suite
def run_sensitivity_suite(base_settings, time_limit=300):
    """
    Run a comprehensive sensitivity analysis on multiple parameters.
    
    Args:
        base_settings (dict): Base parameter settings
        time_limit (int): Time limit for each optimization run
    """
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"sensitivity_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save baseline settings
    with open(f"{output_dir}/baseline_settings.txt", "w") as f:
        for param, value in base_settings.items():
            if isinstance(value, dict) and len(value) > 10:
                f.write(f"{param}: {len(value)} entries\n")
            else:
                f.write(f"{param}: {value}\n")
    
    # Run sensitivity analyses
    
    # 1. Vehicle Capacity (CA)
    capacity_range = [20, 25, 30, 35, 40, 45, 50]
    run_sensitivity_analysis("CA", capacity_range, base_settings, time_limit, output_dir)
    
    # 2. Earliness/Tardiness Penalty (alpha)
    alpha_range = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    run_sensitivity_analysis("alpha", alpha_range, base_settings, time_limit, output_dir)
    
    # 3. Number of Batches
    batch_range = [3, 5, 7, 9, 11]
    run_sensitivity_analysis("num_batches", batch_range, base_settings, time_limit, output_dir)
    
    # 4. Number of Vehicles
    vehicle_range = [4, 6, 8, 10, 12]
    run_sensitivity_analysis("num_vehicles", vehicle_range, base_settings, time_limit, output_dir)
    
    # 5. Temperature Range
    temp_range = [4, 6, 8, 10, 12, 14, 16]
    run_sensitivity_analysis("temperature_range", temp_range, base_settings, time_limit, output_dir)
    
    # 6. Shelf Life Factor
    shelf_life_range = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    run_sensitivity_analysis("shelf_life_factor", shelf_life_range, base_settings, time_limit, output_dir)
    
    # 7. Revenue Factor
    revenue_range = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    run_sensitivity_analysis("revenue_factor", revenue_range, base_settings, time_limit, output_dir)
    
    print(f"\nFull sensitivity analysis suite completed. Results saved to {output_dir}")

# Example usage
if __name__ == "__main__":
    print("Starting sensitivity analysis")
    
    # Create baseline settings
    base_settings = create_baseline_settings(num_orders=15)  # Use fewer orders for faster runs
    
    # Set a time limit for each optimization run (in seconds)
    time_limit = 60
    
    # Run a single sensitivity analysis
    # Example: Analyze sensitivity to vehicle capacity
    capacity_range = [20, 25, 30, 35, 40, 45, 50]
    run_sensitivity_analysis("CA", capacity_range, base_settings, time_limit)
    
    # Uncomment to run full sensitivity suite
    print("\nRunning full sensitivity analysis suite...")
    run_sensitivity_suite(base_settings, time_limit)