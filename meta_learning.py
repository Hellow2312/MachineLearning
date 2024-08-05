import numpy as np

class SimpleLinearModel:
    def __init__(self):
        self.W = np.random.randn(1)
        self.b = np.random.randn(1)

    def forward(self, x):
        return x * self.W + self.b

    def parameters(self):
        return [self.W, self.b]

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def generate_task_data(slope, intercept, num_samples=10):
    x = np.random.rand(num_samples, 1) * 10
    y = slope * x + intercept + np.random.randn(num_samples, 1) * 0.1
    return x, y

def compute_gradients(model, x, y):
    y_pred = model.forward(x)
    loss = mse_loss(y_pred, y)
    dW = np.mean(2 * x * (y_pred - y))
    db = np.mean(2 * (y_pred - y))
    return dW, db, loss

def meta_train(model, num_tasks=1000, inner_lr=0.01, meta_lr=0.001, inner_steps=1, num_samples=10):
    for task in range(num_tasks):
        slope = np.random.rand() * 2 - 1
        intercept = np.random.rand() * 2 - 1
        x, y = generate_task_data(slope, intercept, num_samples)

        W_copy = model.W.copy()
        b_copy = model.b.copy()

        for _ in range(inner_steps):
            dW, db, _ = compute_gradients(model, x, y)
            W_copy -= inner_lr * dW
            b_copy -= inner_lr * db

        model.W -= meta_lr * (W_copy - model.W)
        model.b -= meta_lr * (b_copy - model.b)

        if task % 100 == 0:
            _, _, loss = compute_gradients(model, x, y)
            print(f"Task {task}, Loss: {loss}")

def meta_evaluate(model, num_tasks=5, inner_lr=0.01, inner_steps=10, num_samples=10):
    for task in range(num_tasks):
        slope = np.random.rand() * 2 - 1
        intercept = np.random.rand() * 2 - 1
        x, y = generate_task_data(slope, intercept, num_samples)

        W_copy = model.W.copy()
        b_copy = model.b.copy()

        print(f"\nTask {task + 1}:")
        print(f"True slope: {slope:.4f}, True intercept: {intercept:.4f}")
        print(f"Initial prediction: y = {W_copy[0]:.4f}x + {b_copy[0]:.4f}")

        for step in range(inner_steps):
            dW, db, loss = compute_gradients(model, x, y)
            W_copy -= inner_lr * dW
            b_copy -= inner_lr * db
            if step % 2 == 0:
                print(f"Step {step}, Loss: {loss:.4f}")

        print(f"Final prediction: y = {W_copy[0]:.4f}x + {b_copy[0]:.4f}")

# Inicializar y entrenar el modelo
model = SimpleLinearModel()
meta_train(model)

# Evaluar el modelo en nuevas tareas
meta_evaluate(model)