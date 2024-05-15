##
# (c) 2024 Prof. Dr. sc. hum. Markus Graf
#
# neural network simulating app - tkinter GUI for the simulator
#

import random
import math
import tkinter as tk
import tkinter.ttk as ttk
import time

import neuron


class NNDemonstrator:
    VERSION = "00.01.0001"

    def __init__(self):
        self.running = True
        self.window = tk.Tk()
        self.window.title("NN Demo")
        self.window.geometry("840x780")
        self.window.configure(bg="black")
        self.window.protocol("WM_DELETE_WINDOW", self.on_quit)

        self.description = tk.StringVar()
        self.description.set("Demonstrating Perceptron")

        self.ed_alpha = tk.DoubleVar(value=0.00001)
        self.ed_epsilon = tk.DoubleVar(value=0.2)

        #self.photo = decode_image(icon_image)
        #self.window.iconphoto(False, self.photo)

        self.tab_control = None
        self.canvas = None
        self.canvas_weights = []
        self.menu = None
        self.create_menu()
        self.create_controls()

        # self.training_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
        # self.training_labels = [[0], [0], [0], [1]]

        # with bias (last item = 1)
        self.training_data = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
        self.training_labels = [[0], [0], [0], [1]]

        self.test_inputs = [1, 0, 1]
        self.test_index = 0

        self.layer_specs = [1]
        self.network = neuron.NeuralNetwork( len(self.training_data[0]), layer_specs=self.layer_specs )
        self.network.feed_forward(self.test_inputs)
        self.draw_network()

    def draw_weight(self, neuron_from, neuron_to, weight):
        box_from = self.canvas.coords(neuron_from.item)
        box_to = self.canvas.coords(neuron_to.item)

        if weight < 0.0:
            intensity = int(127*math.fabs(weight))
            if intensity > 127:
                intensity = 127
            color = "#%02x%02x%02x" % (127+intensity, 0, 0)
        else:
            intensity = int(127*math.fabs(weight))
            if intensity > 127:
                intensity = 127
            color = "#%02x%02x%02x" % (0, 127+intensity, 0)

        line_width = max(1, int(math.fabs(weight) * 10))

        self.canvas_weights.append(
            self.canvas.create_line(box_from[2], (box_from[1]+box_from[3])//2,
                                    box_to[0], (box_to[1]+box_to[3])//2, width=line_width,
                                    fill=color)
        )
        self.canvas_weights.append(
            self.canvas.create_text((box_from[2]*0.7+box_to[0]*0.3), (box_from[1]*0.7 + box_to[1]*0.3),
                                    text="{:0.2f}".format(weight),
                                    fill=color)
        )

    def draw_network(self):
        most = 0
        for j in range(0, len(self.network.layers)):
            most = max(most, len(self.network.layers[j]))

        for j in range(0, len(self.network.layers)):
            for i in range(0, len(self.network.layers[j])):
                n = self.network.layers[j][i]
                items = len(self.network.layers[j])
                x = 20 + j*150
                y = 20 + i*150 + (most-items)*75
                if j == 0:
                    n.item = self.canvas.create_rectangle(x, y, x + 50, y + 50,
                                                          fill="green" if n.is_active() else "red")
                    n.text_item = self.canvas.create_text(x+25, y+25, text="{:.2f}".format(n.value()), width=40, fill="black")
                else:
                    n.item = self.canvas.create_oval( x, y, x+50, y+50, fill="green" if n.is_active() else "red")
                    n.text_item = self.canvas.create_text(x+25, y+25, text="{:.2f}".format(n.value()), width=40, fill="black")
                    for k in range(0, len(n.in_neurons)):
                        self.draw_weight(n.in_neurons[k], n, n.weights[k])

    def clear_network(self):
        for j in range(0, len(self.network.layers)):
            for i in range(0, len(self.network.layers[j])):
                n = self.network.layers[j][i]
                self.canvas.delete(n.item)
                self.canvas.delete(n.text_item)
        for i in range(0, len(self.canvas_weights)):
            self.canvas.delete(self.canvas_weights[i])
        self.canvas_weights = []

    def create_menu(self):
        self.menu = tk.Menu(self.window)
        self.window.config(menu=self.menu)
        training_menu = tk.Menu(self.menu)
        self.menu.add_cascade(label="File", menu=training_menu)
        training_menu.add_command(label="Train AND (Perceptron)", command=self.set_training_data_AND_with_bias)
        training_menu.add_command(label="Train OR (Perceptron)", command=self.set_training_data_OR_with_bias)
        training_menu.add_command(label="Train XOR (Perceptron) - will fail", command=self.set_training_data_XOR_with_bias)
        training_menu.add_separator()
        training_menu.add_command(label="Multilayer Perceptron (XOR) working", command=self.set_training_working_XOR)
        training_menu.add_command(label="Deep Learning demo", command=self.set_training_DL)
        # training_menu.add_command(label="Set data dir...", command=self.get_root_path)
        # training_menu.add_command(label="Load previous checkpoint...", command=self.load_state)
        # training_menu.add_command(label="Open test file...", command=self.get_image_filename)
        training_menu.add_separator()
        training_menu.add_command(label="Quit", command=self.on_quit)
        # set about menu
        self.window.createcommand('tkAboutDialog',
                                  lambda: tk.messagebox.showinfo(title="About NeuralNet Demo",
                                                                 message="Version {}\r\n"
                                                                         "(c) 2024 Prof. Dr. Markus Graf".
                                                                 format(NNDemonstrator.VERSION)))
        # self.tk.createcommand('tk::mac::ShowPreferences', self.show_preferences)  # set preferences menu
        # self.tk.createcommand('tk::mac::ShowHelp', self.show_help)  # set help menu

    def create_controls(self):
        self.tab_control = tk.ttk.Notebook(self.window, padding=(5, 5, 5, 5))
        tab_params = tk.ttk.Frame(self.tab_control, padding=(10, 10, 10, 10))
        tab_demo = tk.ttk.Frame(self.tab_control, padding=(10, 10, 10, 10))
        tab_params.grid_columnconfigure(1, weight=1)
        tab_params.grid_rowconfigure(3, weight=1)

        tab_demo.grid_columnconfigure(0, weight=1)
        tab_demo.grid_rowconfigure(0, weight=1)

        self.tab_control.add(tab_params, text="Parameters")
        self.tab_control.add(tab_demo, text="Demo")

        self.window.grid_columnconfigure(0, weight=1)

        tk.Label(tab_params, text="Regularization (alpha): ").grid(column=0, row=0, sticky='NW')
        ed_alpha = tk.Entry(tab_params, textvariable=self.ed_alpha)
        ed_alpha.grid(column=1, row=0, sticky='NWE')
        tk.Label(tab_params, text="Learning Rate (epsilon): ").grid(column=0, row=1, sticky='NW')
        ed_epsilon = tk.Entry(tab_params, textvariable=self.ed_epsilon)
        ed_epsilon.grid(column=1, row=1, sticky='NWE')

        self.canvas = tk.Canvas(tab_demo, width=400, height=400, bg="white")
        self.canvas.grid(column=0, row=0, columnspan=6, sticky='NWSE')

        tk.Label(tab_demo, textvariable=self.description).grid(column=0, row=1, sticky='NW')

        bt_reset = tk.Button(tab_demo, text="Reset Net", command=self.on_reset)
        bt_reset.grid(column=1, row=1, sticky='NW')

        bt_train = tk.Button(tab_demo, text="Train step", command=self.on_train_step)
        bt_train.grid(column=2, row=1, sticky='NW')

        bt_train100 = tk.Button(tab_demo, text="Train 100", command=self.on_train100)
        bt_train100.grid(column=3, row=1, sticky='NW')

        bt_test = tk.Button(tab_demo, text="Test with random", command=self.on_test)
        bt_test.grid(column=4, row=1, sticky='NW')

        bt_test_iterate = tk.Button(tab_demo, text="Test next", command=self.on_test_next)
        bt_test_iterate.grid(column=5, row=1, sticky='NW')

        #self.result = tk.Label(self.window, text="Drag the inner circle to interpolate. Right click to choose color.")
        #self.result.grid(column=0, row=1, sticky='NW')

        self.tab_control.pack(expand=1, fill="both")

    def set_training_data_OR_with_bias(self):
        # with bias (last item = 1)
        self.description.set("Demo OR (with bias neuron)")
        self.training_data = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
        self.training_labels = [[0], [1], [1], [1]]
        self.layer_specs = [1]
        self.on_reset()

    def set_training_data_AND_with_bias(self):
        # with bias (last item = 1)
        self.description.set("Demo AND (with bias neuron)")
        self.training_data = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
        self.training_labels = [[0], [0], [0], [1]]
        self.layer_specs = [1]
        self.on_reset()

    def set_training_data_XOR_with_bias(self):
        # with bias (last item = 1)
        self.description.set("Demo XOR (with bias neuron)")
        self.training_data = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
        self.training_labels = [[0], [1], [1], [0]]
        self.layer_specs = [1]
        self.on_reset()

    def set_training_working_XOR(self):
        self.description.set("Multilayer Perceptron for XOR")
        self.training_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
        self.training_labels = [[0], [1], [1], [0]]

        self.test_inputs = self.training_data[3]
        self.layer_specs = [2, 1]

        self.clear_network()
        self.network = neuron.NeuralNetwork( len(self.training_data[0]),
                                             layer_specs=self.layer_specs,
                                             activation_class=neuron.WeakReLU)
        self.network.feed_forward(self.test_inputs)
        self.network.layers[1][0].weights = [1, -1]
        self.network.layers[1][1].weights = [-1, 1]
        self.network.layers[2][0].weights = [1, 1]
        self.draw_network()

    def set_training_DL(self):
        self.description.set("Deep Neural Network (XOR)")

        self.training_data = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
        self.training_labels = [[0], [1], [1], [0]]

        self.test_inputs = self.training_data[3]
        self.layer_specs = [2, 3, 1]

        self.clear_network()
        self.network = neuron.NeuralNetwork( len(self.training_data[0]),
                                             layer_specs=self.layer_specs,
                                             activation_class=neuron.WeakReLU,
                                             alpha=self.ed_alpha.get(), epsilon=self.ed_epsilon.get())
        self.network.layers[3][0].activation = neuron.Heaviside()
        self.network.feed_forward(self.test_inputs)
        self.draw_network()

    def on_reset(self):
        self.clear_network()
        self.test_inputs = self.training_data[0]
        self.network = neuron.NeuralNetwork( len(self.training_data[0]), layer_specs=self.layer_specs)
        self.network.feed_forward(self.test_inputs)
        self.draw_network()
        self.test_index = 0

    def on_train100(self):
        total_error = 0
        for i in range(0, 100):
            k = int(random.random()*len( self.training_data ))
            inputs = self.training_data[k]
            labels = self.training_labels[k]
            print("{}. index chosen: {}".format(k+1, inputs))
            error = self.network.train_on(inputs, labels)
            total_error += error
            print("[INFO] network training err-fct-result: {}".format(error))
        total_error /= 100
        self.clear_network()
        text = self.canvas.create_text(150, 525, text="Trained 100 - Error {:.6f}".format(total_error), width=340, fill="black")
        self.canvas_weights.append(text)
        self.draw_network()
        self.test_index = 0

    def on_train_step(self):
        k = int(random.random()*len( self.training_data ))
        inputs = self.training_data[k]
        labels = self.training_labels[k]
        error = self.network.train_on(inputs, labels)
        print("[INFO] network training err-fct-result: {}".format(error))
        self.clear_network()
        text = self.canvas.create_text(50, 525, text="Error {:.6f}".format(error), width=140, fill="black")
        self.canvas_weights.append(text)
        self.draw_network()
        self.test_index = 0

    def on_test(self):
        k = int(random.random()*len( self.training_data ))
        inputs = self.training_data[k]
        labels = self.training_labels[k]
        results = self.network.feed_forward(inputs)
        self.clear_network()
        self.draw_network()

    def on_test_next(self):
        inputs = self.training_data[self.test_index]
        labels = self.training_labels[self.test_index]
        results = self.network.feed_forward(inputs)
        error = 0
        for i in range(0, len(labels)):
            error += math.fabs(results[i]-labels[i])
        self.clear_network()
        self.draw_network()

        text = self.canvas.create_text(50, 525, text="Error {:.6f}".format(error), width=140, fill="black")
        self.canvas_weights.append(text)

        self.test_index += 1
        if self.test_index >= len(self.training_data):
            self.test_index = 0

    def on_quit(self):
        self.running = False
        self.window.quit()

    def mainloop(self):
        while self.running:
            self.window.update_idletasks()
            self.window.update()
            # react on when training thread is finished
            # if not self.queue.empty() and self.queue.get_nowait() == 1:
            #    self.update_results()
            time.sleep(0.1)
