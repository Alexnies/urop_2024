import os
import matplotlib.pyplot as plt
import torch


def plot_prediction_comparison(model_load_path: str,
                               X_test: torch.tensor,
                               y_test: torch.tensor,
                               timestamp: str,
                               labels: list):
    # Load the model
    model = RegressionModel(activation1='elu',
                            activation2='elu',
                            activation3='elu',
                            hidden_size=500)
    model.load_state_dict(torch.load(model_load_path))
    model.to("cpu")
    model.eval()

    print(f"The first 10 rows for X_test is: {X_test[:10]}")

    X_test = X_test.to("cpu")
    y_test = y_test.to("cpu")

    with torch.inference_mode():
        y_pred = model(X_test)

    y_test = y_test.numpy()
    y_pred = y_pred.numpy()

    # Plot predictions vs actual values for each feature
    num_features = y_test.shape[1]
    for i in range(num_features):
        plt.figure()
        plt.scatter(y_test[:, i], y_pred[:, i], alpha=0.5)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Comparison for {labels[i]}')

        # save figure
        figures_dir = os.path.join(os.getcwd(), timestamp, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        fig_name = f"{labels[i]}.png"
        save_path = os.path.join(figures_dir, fig_name)
        plt.savefig(save_path, bbox_inches='tight')

        # show figure
        plt.show()

        # close figure
        plt.close()
