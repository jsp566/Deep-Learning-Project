import cProfile
import os
import pstats
from datetime import datetime
import wandb
import src.sweeping as sweeping

def profile_sweep(entity, project, sweep_config, x_train, y_train, x_valid, y_valid):
    filename = project + datetime.now().strftime("%Y%m%d_%H%M%S")
    profiler = cProfile.Profile()
    print(f"{datetime.now()} Profiling {filename}")
    profiler.enable()
    try:
        sweeping.train_sweep(entity, project, config=sweep_config, x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid)
    except Exception as e:
        print(f"{datetime.now()} Error during profiling: {e}")
        raise e
    profiler.disable()
    print(f"{datetime.now()} Profiling {filename} completed")

    save_dir = 'Output/Profiles'
    os.makedirs(save_dir, exist_ok=True)
    profile_name = filename + '_profile.txt'

    with open(os.path.join(save_dir, profile_name), 'w') as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats('cumulative')
        stats.print_stats(50)
        stats.sort_stats('time')
        stats.print_stats(50)
        stats.sort_stats('calls')
        stats.print_stats(100)