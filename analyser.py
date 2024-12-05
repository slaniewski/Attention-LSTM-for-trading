import json
import numpy as np
import matplotlib.pyplot as plt

#namm = "2023-11-17_01-26"
namm = "2024-03-19_04-28"
# Load the JSON file
with open(f'reports/summary/training_results_{namm}.json', 'r') as f:
    results = json.load(f)

# Sort the models based on test_loss and take the top 3
loss_mod_sort = sorted(results.items(), key=lambda x: x[1]['test_loss'])
top_models = loss_mod_sort[:10]
bot_models = loss_mod_sort[-10:]

print("TOP")
for model, details in top_models:
    print(f"Model {model} - Test Loss: {details['test_loss']}")


print("BOT")
for model, details in bot_models:
    print(f"Model {model} - Test Loss: {details['test_loss']}")


# Group by batch
batch_groups = {}
for key, data in results.items():
    batch_name = key.split('_')[1]  # Assuming the batch name is before the first underscore
    if batch_name not in batch_groups:
        batch_groups[batch_name] = []
    batch_groups[batch_name].append((key, data))

# Extract top models from each batch
top_models_per_batch = []
for batch_name, models in batch_groups.items():
    sorted_models = sorted(models, key=lambda x: x[1]['test_loss'])
    top_model = sorted_models[0]  # Get the best model for this batch
    top_models_per_batch.append(top_model)
    top_model = sorted_models[1]  # Get the 2nd best model for this batch
    top_models_per_batch.append(top_model)
    top_model = sorted_models[2]  # Get the 3rd best model for this batch
    top_models_per_batch.append(top_model)

# Extract hyperparameters
hp_values = [details['hyperparameters'] for _, details in top_models_per_batch]

# Plot histograms for each hyperparameter
for hp, _ in hp_values[0].items():
    values = [model_hp[hp] for model_hp in hp_values]
    plt.hist(values, bins=16, alpha=0.75)
    plt.title(f"Histogram for {hp}")
    plt.savefig(f"reports/summary/analysed_hyper_perbatch_{namm}_{hp}.png")
    plt.clf()




# loss_mod_sort = sorted(results.items(), key=lambda x: x[1]['test_loss'])
# # Consider the top half models
# half_len = len(results) // 3
# top_half_models = sorted(results.items(), key=lambda x: x[1]['test_loss'])[:half_len]

# # Extract hyperparameters
# hp_values = [details['hyperparameters'] for _, details in top_half_models]

# # Plot histograms for each hyperparameter
# for hp, _ in hp_values[0].items():
#     values = [model_hp[hp] for model_hp in hp_values]
#     plt.hist(values, bins=16, alpha=0.75)
#     plt.title(f"Histogram for {hp}")
#     plt.savefig(f"reports/summary/analysed_hyper_04-17_{hp}.png")
#     plt.clf()


# Extract batch names
# batches = set([key.split('_trial_')[0] for key in results.keys()])

# # Iterate through batches and compute statistics for test_loss
# for batch in sorted(batches):
#     test_losses = [trial_data['test_loss'] for key, trial_data in results.items() if batch in key]
#     val_loss = [trial_data['final_val_loss'] for key, trial_data in results.items() if batch in key]

#     print(f"Stats for {batch}:")
#     print(f"\tLowest val_loss: {min(val_loss)}")
#     print(f"\tHighest val_loss: {max(val_loss)}")
#     print(f"\tMean val_loss: {np.mean(val_loss)}")
#     print(f"\tStandard deviation: {np.std(val_loss)}")
#     print("--------------------------------------------------")
#     print(f"Stats for {batch}:")
#     print(f"\tLowest test_loss: {min(test_losses)}")
#     print(f"\tHighest test_loss: {max(test_losses)}")
#     print(f"\tMean test_loss: {np.mean(test_losses)}")
#     print(f"\tStandard deviation: {np.std(test_losses)}")
#     print("--------------------------------------------------")
