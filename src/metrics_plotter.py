import matplotlib
matplotlib.use("QtAgg")

import matplotlib.pyplot as plt


class MetricsPlotter:
    def __init__(self):
        plt.style.use("dark_background")

        plt.rcParams["figure.facecolor"] = "#1e1e1e"
        # plt.rcParams["axes.facecolor"] = "#252526"
        plt.rcParams["axes.facecolor"] = "#1e1e1e"
        plt.rcParams["savefig.facecolor"] = "#1e1e1e"

        plt.rcParams["grid.color"] = "#3c3c3c"
        plt.rcParams["grid.alpha"] = 0.6

        plt.rcParams["axes.edgecolor"] = "#808080"
        plt.rcParams["xtick.color"] = "#d4d4d4"
        plt.rcParams["ytick.color"] = "#d4d4d4"
        plt.rcParams["text.color"] = "#d4d4d4"
        plt.rcParams["axes.labelcolor"] = "#d4d4d4"

        self.loss_color = "#4e8cff"
        self.train_color = "#00c29b"
        self.test_color = "#c3890d"

        self.loss_history = []
        self.train_acc_history = []
        self.test_acc_history = []

        plt.ion()

        self.fig, (self.ax1, self.ax2) = (
            plt.subplots(
                1,
                2,
                figsize=(10, 5),
            )
        )

        self.fig.show()

    def update(
        self,
        loss,
        train_accuracy,
        test_accuracy,
    ):
        self.loss_history.append(loss)
        self.train_acc_history.append(train_accuracy)
        self.test_acc_history.append(test_accuracy)

        epochs = range(
            1,
            len(self.loss_history) + 1,
        )

        self.ax1.clear()
        self.ax1.plot(
            epochs,
            self.loss_history,
            color=self.loss_color,
        )
        self.ax1.set_title("Training Loss")
        self.ax1.set_xlabel("Epoch")
        self.ax1.set_ylabel("Loss")
        self.ax1.grid(True)

        self.ax2.clear()
        self.ax2.plot(
            epochs,
            self.train_acc_history,
            label="Train Accuracy",
            color=self.train_color,
        )
        self.ax2.plot(
            epochs,
            self.test_acc_history,
            label="Test Accuracy",
            color=self.test_color,
        )
        self.ax2.set_title("Accuracy")
        self.ax2.set_xlabel("Epoch")
        self.ax2.set_ylabel("Accuracy")
        self.ax2.legend(loc="upper left")
        self.ax2.grid(True)

        self.fig.canvas.draw_idle()
        plt.pause(0.01)

    def process_events(self):
        plt.pause(0.001)
    
    def close(self):
        plt.ioff()
        plt.show()