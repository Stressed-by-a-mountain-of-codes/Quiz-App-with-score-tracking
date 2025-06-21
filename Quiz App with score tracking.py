import tkinter as tk
from tkinter import messagebox
import random

# 25 ML/AI Questions
quiz_data = [
    {"question": "What does 'ML' stand for?", "options": ["Machine Learning", "Model Logic", "Memory Learning", "Meta Language"], "answer": "Machine Learning"},
    {"question": "Which algorithm is used for classification?", "options": ["Linear Regression", "K-Means", "Logistic Regression", "PCA"], "answer": "Logistic Regression"},
    {"question": "Which is an unsupervised algorithm?", "options": ["K-NN", "Decision Tree", "PCA", "Linear Regression"], "answer": "PCA"},
    {"question": "What is deep learning?", "options": ["Subset of ML using neural networks", "Shallow ML technique", "A symbolic AI method", "Non-algorithmic approach"], "answer": "Subset of ML using neural networks"},
    {"question": "What activation function sets negative inputs to zero?", "options": ["Sigmoid", "Tanh", "ReLU", "Softmax"], "answer": "ReLU"},
    {"question": "What is overfitting?", "options": ["Model fits training data too well and fails to generalize", "Model underfits training data", "Training runs too quickly", "Model has too few parameters"], "answer": "Model fits training data too well and fails to generalize"},
    {"question": "Which metric is best for imbalanced classification?", "options": ["Accuracy", "Precision", "Recall", "F1 Score"], "answer": "F1 Score"},
    {"question": "What does 'CNN' stand for?", "options": ["Convolutional Neural Network", "Central Neural Node", "Clustered Network Node", "None"], "answer": "Convolutional Neural Network"},
    {"question": "Which algorithm is used for clustering?", "options": ["Naive Bayes", "K-Means", "Random Forest", "LSTM"], "answer": "K-Means"},
    {"question": "Which model is used for sequential data?", "options": ["Random Forest", "SVM", "RNN", "KNN"], "answer": "RNN"},
    {"question": "Which library is best for deep learning?", "options": ["Pandas", "Scikit-learn", "TensorFlow", "NumPy"], "answer": "TensorFlow"},
    {"question": "What is the role of dropout in neural networks?", "options": ["Increase overfitting", "Speed up learning", "Prevent overfitting", "Add new neurons"], "answer": "Prevent overfitting"},
    {"question": "Which algorithm is lazy learner?", "options": ["Naive Bayes", "KNN", "Decision Tree", "SVM"], "answer": "KNN"},
    {"question": "What is the output of softmax?", "options": ["Binary value", "Real number", "Probability distribution", "Label"], "answer": "Probability distribution"},
    {"question": "Which of the following is a regression algorithm?", "options": ["Decision Tree", "K-Means", "KNN", "Linear Regression"], "answer": "Linear Regression"},
    {"question": "Which algorithm works on distance metric?", "options": ["KNN", "Naive Bayes", "Logistic Regression", "Random Forest"], "answer": "KNN"},
    {"question": "Which loss function is used for classification?", "options": ["Mean Squared Error", "Binary Cross-Entropy", "Huber Loss", "MAE"], "answer": "Binary Cross-Entropy"},
    {"question": "Which technique reduces features?", "options": ["Gradient Descent", "PCA", "Ridge Regression", "Batch Normalization"], "answer": "PCA"},
    {"question": "Which model ensemble method uses voting?", "options": ["Boosting", "Bagging", "Stacking", "Voting Classifier"], "answer": "Voting Classifier"},
    {"question": "What does 'bias' in a model mean?", "options": ["Error due to randomness", "Error from oversimplification", "Overfitting measure", "Model complexity"], "answer": "Error from oversimplification"},
    {"question": "Which ML type uses rewards?", "options": ["Supervised", "Unsupervised", "Reinforcement", "Deep Learning"], "answer": "Reinforcement"},
    {"question": "What is used to split datasets?", "options": ["Scaler", "Normalizer", "train_test_split", "fit_transform"], "answer": "train_test_split"},
    {"question": "Which optimizer is adaptive?", "options": ["SGD", "Adam", "RMSprop", "Adagrad"], "answer": "Adam"},
    {"question": "Which is not a kernel in SVM?", "options": ["Linear", "RBF", "Polynomial", "Convolution"], "answer": "Convolution"},
    {"question": "What does a confusion matrix NOT show?", "options": ["TP", "FP", "Loss", "TN"], "answer": "Loss"}
]

class QuizApp:
    def __init__(self, root, time_limit=15):
        self.root = root
        self.root.title("ML/AI Quiz App")
        self.root.geometry("650x500")

        self.time_limit = time_limit
        self.timer_id = None
        self.remaining_time = time_limit

        random.shuffle(quiz_data)
        self.questions = quiz_data[:25]
        self.current = 0
        self.score = 0
        self.selected_option = tk.StringVar()

        self.header = tk.Label(root, text="🤖 Machine Learning & AI Quiz", font=("Arial", 18, "bold"))
        self.header.pack(pady=10)

        self.q_label = tk.Label(root, text="", font=("Arial", 14), wraplength=600, justify="left")
        self.q_label.pack(pady=10)

        self.radio_buttons = []
        for _ in range(4):
            rb = tk.Radiobutton(root, text="", font=("Arial", 12), variable=self.selected_option, value="")
            rb.pack(anchor="w", padx=30)
            self.radio_buttons.append(rb)

        self.timer_label = tk.Label(root, text=f"⏱ Time left: {self.time_limit} sec", font=("Arial", 12), fg="red")
        self.timer_label.pack(pady=5)

        self.feedback = tk.Label(root, text="", font=("Arial", 12))
        self.feedback.pack(pady=5)

        self.next_btn = tk.Button(root, text="Next", font=("Arial", 12), command=self.next_question)
        self.next_btn.pack(pady=10)

        self.load_question()

    def load_question(self):
        self.reset_timer()
        q = self.questions[self.current]
        self.q_label.config(text=f"Q{self.current + 1}: {q['question']}")
        self.selected_option.set(None)
        for i in range(4):
            self.radio_buttons[i].config(text=q["options"][i], value=q["options"][i])
        self.start_timer()

    def start_timer(self):
        self.remaining_time = self.time_limit
        self.update_timer()

    def update_timer(self):
        self.timer_label.config(text=f"⏱ Time left: {self.remaining_time} sec")
        if self.remaining_time > 0:
            self.remaining_time -= 1
            self.timer_id = self.root.after(1000, self.update_timer)
        else:
            self.feedback.config(text="⏰ Time's up!", fg="red")
            self.after_answer(None)

    def reset_timer(self):
        if self.timer_id:
            self.root.after_cancel(self.timer_id)
            self.timer_id = None

    def next_question(self):
        self.after_answer(self.selected_option.get())

    def after_answer(self, selected):
        self.reset_timer()
        correct = self.questions[self.current]["answer"]
        if selected == correct:
            self.score += 1
            self.feedback.config(text="✅ Correct!", fg="green")
        elif selected is None:
            self.feedback.config(text=f"❌ No answer. Correct: {correct}", fg="red")
        else:
            self.feedback.config(text=f"❌ Incorrect. Correct: {correct}", fg="red")

        self.current += 1
        if self.current < len(self.questions):
            self.root.after(1200, self.load_question)
        else:
            self.root.after(1200, self.show_result)

    def show_result(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        tk.Label(self.root, text="🏁 Quiz Completed!", font=("Arial", 20, "bold")).pack(pady=20)
        tk.Label(self.root, text=f"Score: {self.score} / {len(self.questions)}", font=("Arial", 14)).pack(pady=5)

        percent = (self.score / len(self.questions)) * 100
        status = "🎉 Excellent!" if percent >= 80 else "👍 Good job!" if percent >= 60 else "😐 Keep practicing."

        tk.Label(self.root, text=f"Accuracy: {percent:.2f}%", font=("Arial", 13), fg="blue").pack(pady=5)
        tk.Label(self.root, text=status, font=("Arial", 13), fg="green").pack(pady=10)
        tk.Button(self.root, text="Restart Quiz", font=("Arial", 12), command=self.restart_quiz).pack(pady=10)

    def restart_quiz(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        self.__init__(self.root, self.time_limit)

# Run App
if __name__ == "__main__":
    root = tk.Tk()
    app = QuizApp(root, time_limit=15)
    root.mainloop()
