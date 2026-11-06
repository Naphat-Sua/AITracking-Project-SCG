import matplotlib.animation as animation
import matplotlib.pyplot as plt


class Visualization:
    """Live training dashboard used by Visual.py.

    Shows the input images, the hidden-layer output for those images,
    and the accuracy/loss curves while the network trains.
    """

    def __init__(self, title, dpi=70):
        fig = plt.figure(figsize=(12.8, 7.2), dpi=dpi)
        fig.canvas.manager.set_window_title(title)
        fig.set_facecolor('#FFFFFF')

        self.ax_img = fig.add_subplot(221)
        self.ax_hidden = fig.add_subplot(223)
        self.ax_acc = fig.add_subplot(222)
        self.ax_loss = fig.add_subplot(224)

        self.ax_img.set_title("Input images")
        self.ax_hidden.set_title("Hidden layer output")
        self.ax_acc.set_title("Accuracy (%)")
        self.ax_loss.set_title("Loss")
        self.ax_img.set_axis_off()
        self.ax_hidden.set_axis_off()

        (self.line_acc,) = self.ax_acc.plot([], [], 'b-', label='training')
        (self.line_val_acc,) = self.ax_acc.plot([], [], 'r-', label='validation')
        (self.line_loss,) = self.ax_loss.plot([], [], 'b-', label='training')
        (self.line_val_loss,) = self.ax_loss.plot([], [], 'r-', label='validation')
        self.ax_acc.legend(loc='lower right')
        self.ax_loss.legend(loc='upper right')
        self.ax_acc.set_ylim(0, 110)

        self.fig = fig
        self.accuracy, self.val_accuracy = [], []
        self.loss, self.val_loss = [], []
        self.image = None
        self.image_hidden = None

    def update_accuracy_line(self, accuracy, val_accuracy):
        self.accuracy, self.val_accuracy = accuracy, val_accuracy

    def update_loss_line(self, loss, val_loss):
        self.loss, self.val_loss = loss, val_loss

    def update_image(self, image):
        self.image = image

    def update_image_hidden(self, image):
        self.image_hidden = image

    def _redraw(self):
        if self.image is not None:
            self.ax_img.imshow(self.image, cmap=plt.cm.gray)
        if self.image_hidden is not None:
            self.ax_hidden.imshow(self.image_hidden, cmap=plt.cm.gray)

        self.line_acc.set_data(range(len(self.accuracy)), self.accuracy)
        self.line_val_acc.set_data(range(len(self.val_accuracy)), self.val_accuracy)
        self.line_loss.set_data(range(len(self.loss)), self.loss)
        self.line_val_loss.set_data(range(len(self.val_loss)), self.val_loss)

        if self.accuracy:
            self.ax_acc.set_xlim(0, len(self.accuracy))
            self.ax_loss.set_xlim(0, len(self.loss))
            self.ax_loss.set_ylim(0, max(max(self.loss), 0.1) * 1.1)
        return self.line_acc, self.line_val_acc, self.line_loss, self.line_val_loss

    def train(self, training_model, iterations, save_movie=False):
        def animate_func(step):
            if step >= iterations - 1:
                print("\n+++++++++++++++++ Finish training +++++++++++++++++")
                plt.close()
            else:
                training_model(step_visual=step, visual=self)
                plt.pause(0.001)  # makes the UI a little more responsive
            return self._redraw()

        ani = animation.FuncAnimation(self.fig, animate_func, iterations,
                                      repeat=False, interval=16, blit=False)
        if save_movie:
            # save all frames to an mp4 file
            mywriter = animation.FFMpegWriter(
                fps=24, codec='libx264',
                extra_args=['-pix_fmt', 'yuv420p', '-profile:v', 'high',
                            '-tune', 'animation', '-crf', '18'])
            ani.save("mnist_training.mp4", writer=mywriter)
        else:
            plt.show()
        return ani
