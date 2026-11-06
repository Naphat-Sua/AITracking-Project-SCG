# From: 
# http://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html
# http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html#sphx-glr-auto-examples-svm-plot-iris-py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tensorflow.keras import Input, optimizers
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential

# my modules
from training_history import TrainingHistory

# Two interleaved groups of 2D points; the network learns to separate them.
X_TRAIN = [[-0.4326, 1.1909], 
	[3.0, 4.0],
	[0.1253 , -0.0376   ],
	[0.2877 ,   0.3273  ],
	[-1.1465 ,   0.1746 ],
	[1.8133 ,   1.0139  ],
	[2.7258 ,   1.0668  ],
	[1.4117 ,   0.5593  ],
	[4.1832 ,   0.3044  ],
	[1.8636 ,   0.1677  ],
	[0.5 ,   3.2  ],
	[0.8 ,   3.2  ],
	[1.0 ,   -2.2  ],
	[2.1 ,   -4.2  ],
	[3.5 ,   -4.7  ],
	[3.0 ,   -5.0  ]]

Y_TRAIN = [ 1 , 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]


def load_dataset():
	"""Return the (X, Y) toy dataset as numpy arrays."""
	return np.array(X_TRAIN), np.array(Y_TRAIN)

def build_MLP(features):
	model = Sequential()
	model.add(Input(shape=(features,)))
	model.add(Dense(units=6))
	model.add(Activation("tanh"))
	model.add(Dense(units=1))
	# now model.output_shape == (None, 1)
	# note: `None` is the batch dimension.
	#
	model.add(Activation("sigmoid"))

	# algorithim to optimize the models (train model)
	# compute loss with function: binary crossentropy
	#opt = optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
	opt = optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	model.compile(optimizer=opt,
				loss='binary_crossentropy',
				metrics=['accuracy'])
	return model

def make_trainer(X_train, Y_train, history):
	"""Build the per-step training callback used by Visualization.train()."""
	def training_model(model, step_visual=0, visual=None):
		model.fit(X_train, Y_train, epochs=10, verbose=0, callbacks=[history])
		if visual is not None:
			visual.update_line(history.loss, history.accuracy)

		iterator = ((step_visual + 1) * 10)
		if iterator % 50 == 0:
			print("============= Iterator %d ================" % iterator)
			# evaluate after trained
			scores = model.evaluate(X_train, Y_train, verbose=0)
			print("Evaluate model: %s = %.4f" % (model.metrics_names[0], scores[0]))
			print("Evaluate model: %s = %.4f" % (model.metrics_names[1], scores[1] * 100))
		return model
	return training_model
		
class Visualization():	
	def __init__(self, model, X_train, Label_train, title, dpi=70):
		fig = plt.figure(figsize=(19.20,10.80), dpi=dpi)
		fig.canvas.manager.set_window_title(title)
		fig.set_facecolor('#FFFFFF')
		ax1 = fig.add_subplot(121)
		ax2 = fig.add_subplot(222)		
		ax3 = fig.add_subplot(224)
		ax1.grid(False) # toggle grid off
		ax2.grid(False) # toggle grid off
		ax3.grid(False) # toggle grid off
								
		self.ax1, self.ax2, self.ax3 = ax1, ax2, ax3		
		self.fig =fig
		
		self.model = model		
		self.xy_label = []		
		self.loss = []
		self.accuracy = []
				
		# There are 2 classess
		class_label = range(0, max(Label_train)+1)
		self.scatter_list = [ self.ax1.scatter([], []) for i in range(0,len(class_label))]
				
		# seperate 2 class of X_train
		for number in class_label:
			index_label = np.where(Label_train == number)[0]
			xy = X_train[index_label]		
			self.xy_label.append(xy)
			
		h = .3  # step size in the mesh
		# create a mesh to plot in
		x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
		y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
				
		self.xx = xx
		self.yy = yy
		
		self.line2, = ax2.plot([], [])		
		self.line3, = ax3.plot([], [])		
				
		ax1.set_xticks(())
		ax1.set_yticks(())
		ax1.set_xlabel('X values')
		ax1.set_ylabel('Y values')
		ax1.set_xlim(xx.min(), xx.max())
		ax1.set_ylim(yy.min(), yy.max())
		
		self.ax1.set_title("Classify 2 groups")
		self.ax2.set_title("Loss")		
		self.ax3.set_title("Accuracy")
		self.ax2.set_ylim(0, 1)   # not autoscaled 	
		self.ax3.set_ylim(0, 110)  # not autoscaled		
		plt.tight_layout()		
		
	def init(self):	
		return self.scatter_list, self.line2, self.line3
			
	def update(self):
		m = ['v', 's']
		colors = ['navy', 'orangered' ]
		
		# Put the result into a color plot
		# (use self.model, which train() keeps up to date, not a global)
		Z = self.model.predict(np.c_[self.xx.ravel(), self.yy.ravel()], verbose=0)
		Z = Z.reshape(self.xx.shape)
		con1 = self.ax1.contourf(self.xx, self.yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)	
			
		for index, data in enumerate(self.xy_label):		
			self.scatter_list[index] = self.ax1.scatter(data[:, 0], data[:, 1], c=colors[index] , marker=m[index])
			#self.scatter_list[index] = self.ax1.scatter(data[:, 0], data[:, 1], cmap=plt.cm.coolwarm , marker=m[index])
		
		# rescale x		
		x2 = range(0, len(self.loss))		
		x3 = range(0, len(self.accuracy))				
		self.ax2.set_xlim(0, max(x2)+1)
		self.ax3.set_xlim(0, max(x3)+1)
		# plot loss and accuracy
		self.line2.set_data(x2, self.loss)
		self.line3.set_data(x3, self.accuracy)	
		return self.scatter_list, con1, self.line2, self.line3
	
	def update_line(self, loss, accuracy):
		self.loss = loss
		self.accuracy = accuracy
		
	def train(self, training_model, iterations, save_movie=False):	
		def animate_func(step):	
			if step >= iterations-1:				 				
				print("\n+++++++++++++++++ Finish training +++++++++++++++++")
				plt.close()			
			else:								
				self.model = training_model(self.model, step_visual=step, visual=self)						
				plt.pause(0.501) # makes the UI a little more responsive 
			return self.update()
	
		ani = animation.FuncAnimation(self.fig, animate_func, iterations, 
						init_func=self.init, repeat=False, interval=16, blit=False)
		if save_movie:
			# save all picture to mp4 file
			mywriter = animation.FFMpegWriter(fps=24, codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-profile:v', 'high', '-tune', 'animation', '-crf', '18'])		
			ani.save("example_2_class.mp4", writer=mywriter)
		else:
			plt.show()

def main():
	X_train, Y_train = load_dataset()
	model = build_MLP(X_train.shape[1])
	history = TrainingHistory()
	training_model = make_trainer(X_train, Y_train, history)

	visual = Visualization(model, X_train, Y_train,
					title="Example: binary classification")
	visual.train(training_model, iterations=70, save_movie=False)


if __name__ == '__main__':
	main()
