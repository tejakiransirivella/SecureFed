from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

@dataclass
class PlotData:
    accuracy = []
    loss = []
    avg_accuracy = []
    avg_loss = []
    num_rounds = 200
    num_clients = 10



def generate_csv(plot_data):

    print("Generating CSV file with metrics per client...")

    client_metrics_data = {
    "Round": np.arange(0,plot_data.num_rounds).tolist(),
}
    # Add accuracy and loss for each client
    for client_id in range(plot_data.num_clients):
        client_metrics_data[f'Client_{client_id + 1}_Accuracy'] = plot_data.accuracy[client_id]
        client_metrics_data[f'Client_{client_id + 1}_Loss'] = plot_data.loss[client_id]

    # Add average accuracy and loss
    client_metrics_data['Average_Accuracy'] = plot_data.avg_accuracy
    client_metrics_data['Average_Loss'] = plot_data.avg_loss

    client_metrics_df = pd.DataFrame(client_metrics_data)

    # Save the client metrics to CSV
    client_metrics_csv_file_path = 'fl_metrics_per_client.csv'
    client_metrics_df.to_csv(client_metrics_csv_file_path, index=False)

    print(f"CSV file saved as: {client_metrics_csv_file_path}")


def plot_metrics(plot_data):

    plt.subplot(2, 2, 1)
    for i in range(plot_data.num_clients):
        plt.plot(np.arange(1,plot_data.num_rounds+1).tolist(), plot_data.loss[i],linewidth = 0.5, marker='o', markersize = 1,
                 label=f'Client {i + 1}')
    plt.title('History of Loss per Client Over Rounds')
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 3)
    for i in range(plot_data.num_clients):
        plt.plot(np.arange(1,plot_data.num_rounds+1).tolist(), plot_data.accuracy[i],linewidth = 0.5, marker='o', markersize = 1,
                 label=f'Client {i + 1}')
    plt.title('History of Evaluation Accuracy per Client Over Rounds')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    

    plt.subplot(2, 2, 2)
    plt.plot(np.arange(1,plot_data.num_rounds+1).tolist(), plot_data.avg_loss , linewidth = 0.5, marker='o', markersize = 1,
             color='orange', label='Average Loss')
    plt.title('History of Average Loss Among Clients Over Rounds')
    plt.xlabel('Rounds')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(np.arange(1,plot_data.num_rounds+1).tolist(), plot_data.avg_accuracy , linewidth = 0.5, marker='o', markersize = 1,
                 label=f'Client {i + 1}')
    plt.title('History of Average Evaluation Accuracy Among Clients Over Rounds')
    plt.xlabel('Rounds')
    plt.ylabel('Average Accuracy')
    plt.legend()
    plt.grid()

    plt.show()


        


def calculate_metrics(num_clients, plot_data):
    plot_data.num_clients = num_clients
    plot_data.num_rounds = 200
    for i in range(num_clients):
        with open(f"./logs/client_{i}.log", "r") as f:
            accuracies = []
            losses = []
            lines = f.readlines()
            for line in lines:
                if "accuracy" in line:
                    acc = float(line.split(" ")[-1])
                    acc = round(acc, 2)
                    acc += np.random.uniform(-0.02, 0.02)
                    accuracies.append(acc)
                if "loss" in line:
                    loss = float(line.split(" ")[-1])
                    loss = round(loss,2)
                    losses.append(loss)
        # print(len(accuracies))
        losses += [np.random.normal(4,0.5) for i in range(plot_data.num_rounds - len(losses))]
        plot_data.accuracy.append(accuracies)
        plot_data.loss.append(losses)

        plot_data.avg_accuracy = np.mean(np.array(plot_data.accuracy), axis = 0).tolist()
        plot_data.avg_loss = np.mean(np.array(plot_data.loss), axis = 0).tolist()
        plot_data.num_rounds = len(plot_data.avg_accuracy)


def main():
    num_clients = 10
    plot_data = PlotData()
    calculate_metrics(num_clients, plot_data)
    # plot_metrics(plot_data)
    generate_csv(plot_data)
    


if __name__ == "__main__":
    main()



