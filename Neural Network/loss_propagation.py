from Data_Generation import *

results_dir = os.path.join(os.path.dirname(abs_path), 'Best_Propagation')

# Create the 'Results' directory if it doesn't exist
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

test_loss_function1, history_function1 = train_and_evaluate((1,),\
                                            x1_train, y1_train, x1_test, y1_test, x1_val, y1_val, 5,
                                            500, 60)

test_loss_function2, history_function2 = train_and_evaluate((2,),\
                                            x2_train, y2_train, x2_test, y2_test, x2_val, y2_val, 7,
                                            500, 100)

print('Test 1 :', test_loss_function1)
print('Test 2 :', test_loss_function2)

# Create a figure for the best case histories
plt.figure()
plt.plot(list(range(1, 500 + 1)), history_function1.history['loss'], label='Training loss')
plt.plot(list(range(1, 500 + 1)), history_function1.history['val_loss'], label='Validation loss')
plt.title("Best Training Progress (Function 1)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.savefig(os.path.join(results_dir, "best_training_plot_function1.png"))
plt.close()

plt.figure()
plt.plot(list(range(1, 500 + 1)), history_function2.history['loss'], label='Training loss')
plt.plot(list(range(1, 500 + 1)), history_function2.history['val_loss'], label='Validation loss')
plt.title("Best Training Progress (Function 2)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.savefig(os.path.join(results_dir, "best_training_plot_function2.png"))
plt.close()