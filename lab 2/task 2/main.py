import numpy as np

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights  # Синаптичні ваги
        self.bias = bias        # Поріг

    def activate(self, inputs):
        # Сумуємо входи і застосовуємо функцію активації
        total_input = np.dot(self.weights, inputs) + self.bias
        return 1 if total_input > 0 else 0  # Повертаємо 1 або 0 в залежності від порогу

class Layer:
    def __init__(self, neurons):
        self.neurons = neurons  # Нейрони в шарі

    def forward(self, inputs):
        # Обчислюємо виходи всіх нейронів шару
        return [neuron.activate(inputs) for neuron in self.neurons]

class NeuralNetwork:
    def __init__(self):
        # Створення нейронів для кожного шару
        self.layer1 = Layer([
            Neuron(weights=[1, 1], bias=-0.5),  # Нейрон для OR
            Neuron(weights=[1, 1], bias=-1.5)   # Нейрон для AND
        ])
        self.layer2 = Neuron(weights=[1, -1], bias=-0.5)  # Нейрон для XOR

    def forward(self, x1, x2):
        # Пряме поширення сигналу через мережу
        inputs = np.array([x1, x2])  # Входи
        layer1_outputs = self.layer1.forward(inputs)  # Виходи першого шару
        return self.layer2.activate(layer1_outputs)  # Вихід з другого шару

def test_neural_network(nn):
    # Тестування всіх можливих входів для XOR
    test_cases = [
        (0, 0, 0),  # Очікується 0
        (0, 1, 1),  # Очікується 1
        (1, 0, 1),  # Очікується 1
        (1, 1, 0)   # Очікується 0
    ]

    for x1, x2, expected in test_cases:
        result = nn.forward(x1, x2)  # Отримуємо результат
        assert result == expected, f"Помилка на вході ({x1}, {x2}): очікував {expected}, отримав {result}"

    print("Всі тести пройшли успішно!")

# Запуск тестів
nn = NeuralNetwork()
test_neural_network(nn)
