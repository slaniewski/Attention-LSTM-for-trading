import json
with open('reports/model_hyperparams_2023-10-29_18-02.json', 'r') as f:
    models   = json.load(f)

value_counts = {}

value_counts = {}

# Count the occurrences of each value for each key
for model in models:
    # Check if model is a dictionary
    if not isinstance(model, dict):
        print(f"Unexpected item (not a dictionary): {model}")
        continue
    for key, value in model.items():
        if key not in value_counts:
            value_counts[key] = {}
        if value not in value_counts[key]:
            value_counts[key][value] = 0
        value_counts[key][value] += 1

# Print the counts
for key, counts in value_counts.items():
    print(f"\n{key}:")
    for value, count in counts.items():
        print(f"  {value}: {count}")

# # Count the occurrences of each value for each key
# for model in models:
#     for key, value in model.items():
#         if key not in value_counts:
#             value_counts[key] = {}
#         if value not in value_counts[key]:
#             value_counts[key][value] = 0
#         value_counts[key][value] += 1

# # Print the counts
# for key, counts in value_counts.items():
#     print(f"\n{key}:")
#     for value, count in counts.items():
#         print(f"  {value}: {count}")