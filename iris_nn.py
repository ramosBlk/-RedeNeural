"""Rede neural NumPy para classificação do conjunto de dados Iris."""

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DatasetSplit:
    """Conjuntos de treino e teste após o pré-processamento."""

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


class IrisDataLoader:
    """Carrega, divide e prepara as amostras do conjunto Iris."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42) -> None:
        self.test_size = test_size
        self.random_state = random_state

    def load(self) -> DatasetSplit:
        iris = load_iris()
        x_train, x_test, labels_train, labels_test = train_test_split(
            iris.data,
            iris.target,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=iris.target,
        )

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        class_count = len(iris.target_names)
        y_train = np.eye(class_count, dtype=np.float64)[labels_train]
        y_test = np.eye(class_count, dtype=np.float64)[labels_test]

        return DatasetSplit(x_train, x_test, y_train, y_test)


class NeuralNetwork:
    """Rede totalmente conectada com uma camada ReLU e saída Softmax."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        random_state: int = 42,
    ) -> None:
        rng = np.random.default_rng(random_state)
        self.weights1 = rng.standard_normal((input_size, hidden_size)) * np.sqrt(
            2.0 / input_size
        )
        self.bias1 = np.zeros((1, hidden_size), dtype=np.float64)
        self.weights2 = rng.standard_normal((hidden_size, output_size)) * np.sqrt(
            2.0 / hidden_size
        )
        self.bias2 = np.zeros((1, output_size), dtype=np.float64)

    @staticmethod
    def _relu(values: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, values)

    @staticmethod
    def _relu_derivative(values: np.ndarray) -> np.ndarray:
        return (values > 0.0).astype(np.float64)

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted_values = values - np.max(values, axis=1, keepdims=True)
        exp_values = np.exp(shifted_values)
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def _forward(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hidden_linear = features @ self.weights1 + self.bias1
        hidden_activation = self._relu(hidden_linear)
        output_linear = hidden_activation @ self.weights2 + self.bias2
        probabilities = self._softmax(output_linear)
        return hidden_linear, hidden_activation, probabilities

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        _, _, probabilities = self._forward(features)
        return probabilities

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(features), axis=1)

    def loss(self, features: np.ndarray, targets: np.ndarray) -> float:
        probabilities = self.predict_proba(features)
        clipped = np.clip(probabilities, 1e-12, 1.0)
        return float(-np.sum(targets * np.log(clipped)) / features.shape[0])

    def train_step(
        self, features: np.ndarray, targets: np.ndarray, learning_rate: float
    ) -> None:
        hidden_linear, hidden_activation, probabilities = self._forward(features)
        sample_count = features.shape[0]

        output_gradient = (probabilities - targets) / sample_count
        weights2_gradient = hidden_activation.T @ output_gradient
        bias2_gradient = np.sum(output_gradient, axis=0, keepdims=True)

        hidden_gradient = output_gradient @ self.weights2.T
        hidden_gradient *= self._relu_derivative(hidden_linear)
        weights1_gradient = features.T @ hidden_gradient
        bias1_gradient = np.sum(hidden_gradient, axis=0, keepdims=True)

        self.weights1 -= learning_rate * weights1_gradient
        self.bias1 -= learning_rate * bias1_gradient
        self.weights2 -= learning_rate * weights2_gradient
        self.bias2 -= learning_rate * bias2_gradient


class Trainer:
    """Executa o gradiente descendente sobre a rede fornecida."""

    def __init__(self, model: NeuralNetwork, learning_rate: float = 0.05) -> None:
        self.model = model
        self.learning_rate = learning_rate

    def fit(
        self, features: np.ndarray, targets: np.ndarray, epochs: int = 1000
    ) -> list[float]:
        losses = []
        for _ in range(epochs):
            self.model.train_step(features, targets, self.learning_rate)
            losses.append(self.model.loss(features, targets))
        return losses


class Evaluator:
    """Calcula métricas de classificação para um modelo treinado."""

    @staticmethod
    def accuracy(
        model: NeuralNetwork, features: np.ndarray, targets: np.ndarray
    ) -> float:
        predicted_labels = model.predict(features)
        expected_labels = np.argmax(targets, axis=1)
        return float(np.mean(predicted_labels == expected_labels))


def main() -> None:
    dataset = IrisDataLoader().load()
    model = NeuralNetwork(input_size=4, hidden_size=8, output_size=3)
    trainer = Trainer(model, learning_rate=0.05)
    losses = trainer.fit(dataset.x_train, dataset.y_train, epochs=1000)

    test_accuracy = Evaluator.accuracy(model, dataset.x_test, dataset.y_test)
    print(f"Perda inicial: {losses[0]:.4f}")
    print(f"Perda final: {losses[-1]:.4f}")
    print(f"Acurácia no teste: {test_accuracy:.2%}")


if __name__ == "__main__":
    main()
