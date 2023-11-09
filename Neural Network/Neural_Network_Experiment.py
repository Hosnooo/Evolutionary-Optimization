from Data_Generation import *

# Perform experiments
hidden_nodes_list = [3, 5, 7, 10]  # Test different numbers of hidden nodes
num_training_samples_list = [20, 40, 60, 80, 100]  # Test different numbers of training samples
num_epochs_def = 500  # Define the default number of epochs

# Experiment 1: Vary hidden nodes, fix samples
results_hidden_nodes_function1 = []
history_hidden_nodes_function1 = []

results_hidden_nodes_function2 = []
history_hidden_nodes_function2 = []

for hidden_nodes in hidden_nodes_list:
    test_loss_function1, history_function1 = train_and_evaluate((1,),\
                                            x1_train, y1_train, x1_test, y1_test, x1_val, y1_val, hidden_nodes,
                                            num_epochs_def, 50)
    results_hidden_nodes_function1.append((hidden_nodes, test_loss_function1[0], test_loss_function1[1]))
    history_hidden_nodes_function1.append(history_function1)

    test_loss_function2, history_function2 = train_and_evaluate((2,),\
                                            x2_train, y2_train, x2_test, y2_test, x2_val, y2_val, hidden_nodes,
                                            num_epochs_def, 50)
    results_hidden_nodes_function2.append((hidden_nodes, test_loss_function2[0], test_loss_function2[1]))
    history_hidden_nodes_function2.append(history_function2)

# Experiment 2: Vary samples, fix hidden nodes
results_samples_function1 = []
history_samples_function1 = []

results_samples_function2 = []
history_samples_function2 = []

for num_samples in num_training_samples_list:
    test_loss_function1, history_function1 = train_and_evaluate((1,), x1_train, y1_train, x1_test, y1_test, x1_val, y1_val, 3,
                                            num_epochs_def, num_samples)
    results_samples_function1.append((num_samples, test_loss_function1[0], test_loss_function1[1]))
    history_samples_function1.append(history_function1)

    test_loss_function2, history_function2 = train_and_evaluate((2,), x2_train, y2_train, x2_test, y2_test, x2_val, y2_val, 3,
                                            num_epochs_def, num_samples)
    results_samples_function2.append((num_samples, test_loss_function2[0], test_loss_function2[1]))
    history_samples_function2.append(history_function2)

# Save results to Excel
save_results_to_excel(results_hidden_nodes_function1, "hidden_nodes_function1_results.xlsx",\
                       ["Num Hidden Nodes", "MSE Loss", "MAPE Loss"])
save_results_to_excel(results_hidden_nodes_function2, "hidden_nodes_function2_results.xlsx",\
                       ["Num Hidden Nodes", "MSE Loss", "MAPE Loss"])

save_results_to_excel(results_samples_function1, "samples_function1_results.xlsx", ["Num Samples", "MSE Loss", "MAPE Loss"])
save_results_to_excel(results_samples_function2, "samples_function2_results.xlsx", ["Num Samples", "MSE Loss", "MAPE Loss"])

# Create figures for hidden nodes
plot_results(hidden_nodes_list, [result[1] for result in results_hidden_nodes_function1],\
              "Test Loss vs. Hidden Nodes (Function 1)", "Num Hidden Nodes", "Test Loss", "hidden_nodes_effect_function1.png")
plot_results(hidden_nodes_list, [result[1] for result in results_hidden_nodes_function2],\
              "Test Loss vs. Hidden Nodes (Function 2)", "Num Hidden Nodes", "Test Loss", "hidden_nodes_effect_function2.png")

# Create figures for training samples
plot_results(num_training_samples_list, [result[1] for result in results_samples_function1],\
              "Test Loss vs. Training Samples (Function 1)", "Num Samples", "Test Loss", "samples_effect_function1.png")
plot_results(num_training_samples_list, [result[1] for result in results_samples_function2],\
              "Test Loss vs. Training Samples (Function 2)", "Num Samples", "Test Loss", "samples_effect_function2.png")