Neural Mycologist: Bio-Data Prediction Engine
A real-time neural network visualizer and predictor built with Pygame and NumPy. This engine simulates the logical processes of a biological neural network as it analyzes environmental factors to predict the occurrence of specific fungal species.
🧠 Neural Architecture
The engine utilizes a classic Multi-Layer Perceptron (MLP) structure:
Input Layer (6 Neurons): Processes raw environmental data: Night/Day Temperature, Humidity, Precipitation, Month, and Latitude.
Hidden Layer (14 Neurons): Perceptive associations. This layer identifies non-linear combinations (e.g., "Sharp temperature drop + High humidity").
Output Layer (1 Neuron): Final probability verdict using a Sigmoid activation function.
🔬 Core Features
Dynamic Synaptic Mapping: Real-time visualization of synaptic threads. The brightness and color of each connection represent the strength and polarity (positive/negative) of the signal passing through.
Cognitive Log: A live text stream translating the network's numerical weights into human-readable "thoughts," explaining which factors currently dominate the prediction.
Pre-trained Logic: The model is pre-trained using Backpropagation on historical environmental datasets to recognize peak seasonal conditions.
Interactive Simulation: Users can manipulate climate variables on the fly to see how the "mycelium resonance" reacts to environmental shifts.
🛠 Tech Stack
Python 3.x
NumPy (Linear Algebra & Matrix Math)
Pygame (Rendering & UI)
🎮 Interface Navigation
Numbers/Period: Input data for the active parameter.
ENTER: Confirm value and cycle to the next input node.
Visual Grid: Technical blueprint overlay for precise spatial observation of neuron activity.
📊 Biological Calibration
The model is specifically tuned for temperate climate zones, prioritizing:
Thermal Induction: Recognizing the stimulation effect of nocturnal cooling.
Hydrological Saturation: Calculating the impact of cumulative precipitation on hyphal activation.
Seasonal Temporal Windows: Analyzing the 12-month cycle to identify peak biological activity windows.
