import json

with open("config/network_metadata.json") as f:
    network = json.load(f)

print(network["assets"]["transformers"][0])
print()
print(network["assets"]["regulators"][0])
print()
print(network["assets"]["capacitors"][0])