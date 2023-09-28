import csv
import datetime

""""Run results for
1. Average fitness of the population each generation
2. Highest fitness value of each generation
3. Sigma value of average system at start generation
4. Sigma value of average system at last generation"""
def save_run(mean_f, best_f, sig_start, sig_end):
    # giving the file a timestamp
    time = datetime.datetime.now().strftime("%d%H%M")
    filename = f'evcomprun_{time}.csv'

    with open(filename, mode='w', newline='') as file:
        f = csv.writer(file)
        f.writerow(['Mean fitness', 'Best fitness', 'Average sigma start', 'Average sigma end'])
        f.writerow([mean_f[0], best_f[0], sig_start, sig_end])

        for i in range(1, len(mean_f)):
            f.writerow([mean_f[i], best_f[i]])
