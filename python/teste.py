import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics


# df = pd.read_csv("resultados.txt")

num_tests = 10
num_threads = 7

avg_time, threads = statistics.calc_avg_time(num_tests, num_threads, "D:/Codes/PCD/TrabalhoFinal/parte1/results1.txt")

df = pd.DataFrame(avg_time, columns=["Time"])
df["NumThreads"] =  threads


speedup = [(df.loc[0]["Time"]/x) for x in df["Time"].to_list()]

df["Speedup"] = speedup

eficiency = [(x/y) for x, y in zip(df["Speedup"].to_list(), df["NumThreads"].to_list())]

df["Eficiency"] = eficiency
print(df.to_string())

def draw_efficiency():
    x_axis = df["NumThreads"].to_list()
    # x_axis = threads
    y_axis = df["Eficiency"].to_list()


    plt.plot(x_axis, y_axis, label="Calculated Eficiency")
    plt.xlabel("Num of Threads")
    plt.ylabel("Eficiency")
    plt.title("Gráfico de Eficiencia")
    plt.legend()
    plt.show()

def draw_speedup():
    x_axis = df["NumThreads"].to_list()
    # x_axis = threads
    y_axis = df["Speedup"].to_list()

    #linear speedup

    lin_x = x_axis
    lin_y = lin_x

    plt.plot(lin_x, lin_y, label="Linear Speedup")
    plt.plot(x_axis, y_axis, label="Calculated Speedup")
    plt.xlabel("Num of Threads")
    plt.ylabel("Speedup")
    plt.title("Gráfico de speedup")
    plt.legend()
    plt.show()


draw_speedup()
draw_efficiency()