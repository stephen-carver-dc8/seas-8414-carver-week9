# Filename: 1_generate_dga_data.py
import csv
import random
import math

def get_entropy(s):
    p, lns = {}, float(len(s))
    for c in s: p[c] = p.get(c, 0) + 1
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())

# Create sample data
header = ['domain', 'length', 'entropy', 'class']
data = []

# Legitimate domains
legit_domains = ['google', 'facebook', 'amazon', 'github', 'wikipedia', 'microsoft']
for _ in range(100):
    domain = random.choice(legit_domains) + ".com"
    data.append([domain, len(domain), get_entropy(domain), 'legit'])

# DGA domains
for _ in range(100):
    length = random.randint(15, 25)
    domain = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(length)) + ".com"
    data.append([domain, len(domain), get_entropy(domain), 'dga'])

with open('dga_dataset_train.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)

print("dga_dataset_train.csv created successfully.")