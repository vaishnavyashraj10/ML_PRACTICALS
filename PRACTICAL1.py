# Practical 1: Mean, Median and Mode

import statistics

data = [10, 20, 20, 30, 40, 50, 20]

mean = statistics.mean(data)
median = statistics.median(data)
mode = statistics.mode(data)

print("Dataset:", data)
print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)