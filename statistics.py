import pandas as pd

def calc_avg_time(num_of_tests, num_threads, file_path):
    avg_thread_count_time = [0 for i in range(num_threads)]

    df = pd.read_csv(file_path)

    threads = df["NumThreads"].to_list()

    for i in range(num_of_tests):
        df_temp = df[i*num_threads:num_threads*(i+1)]    
        avg_thread_count_time = [sum(x) for x in zip(avg_thread_count_time, df_temp["Time"].to_list())]

    avg_thread_count_time = [x/num_of_tests for x in avg_thread_count_time]
    return avg_thread_count_time, threads[:num_threads]



        
