import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class MathScorePredictor:
    def __init__(self, master):
        # Create main window
        self.master = master
        master.title("Math Score Predictor")
        master.geometry("400x300")
        master.configure(bg='#f0f0f0')

        # Create and set up title
        self.title = tk.Label(master, text="Math Score Predictor", 
                               font=("Arial", 16, "bold"), 
                               bg='#f0f0f0', 
                               fg='#333333')
        self.title.pack(pady=20)

        # Create input frames
        self.create_input_fields()

        # Create predict button
        self.predict_button = tk.Button(master, 
                                        text="Predict Math Score", 
                                        command=self.predict_score,
                                        bg='#4CAF50', 
                                        fg='white', 
                                        font=("Arial", 12))
        self.predict_button.pack(pady=20)

        # Result label
        self.result_label = tk.Label(master, 
                                     text="", 
                                     font=("Arial", 14), 
                                     bg='#f0f0f0')
        self.result_label.pack(pady=10)

        # Train the model when app starts
        self.train_model()

    def create_input_fields(self):
        # Create input frames
        input_frame = tk.Frame(self.master, bg='#f0f0f0')
        input_frame.pack(pady=10)

        # Reading Score
        self.reading_label = tk.Label(input_frame, text="Reading Score:", 
                                      bg='#f0f0f0', 
                                      font=("Arial", 12))
        self.reading_label.grid(row=0, column=0, padx=10, pady=5)
        self.reading_entry = tk.Entry(input_frame, font=("Arial", 12), width=10)
        self.reading_entry.grid(row=0, column=1, padx=10, pady=5)

        # Writing Score
        self.writing_label = tk.Label(input_frame, text="Writing Score:", 
                                      bg='#f0f0f0', 
                                      font=("Arial", 12))
        self.writing_label.grid(row=1, column=0, padx=10, pady=5)
        self.writing_entry = tk.Entry(input_frame, font=("Arial", 12), width=10)
        self.writing_entry.grid(row=1, column=1, padx=10, pady=5)

    def train_model(self):
        # Read the dataset
        df = pd.read_csv(r"C:\Users\K_har\Downloads\linear regression\StudentsPerformance.csv")

        # Separate features and target
        X = df[['reading score', 'writing score']]
        y = df['math score']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train the model
        self.model = LinearRegression()
        self.model.fit(X_train_scaled, y_train)

    def predict_score(self):
        try:
            # Get input values
            reading_score = float(self.reading_entry.get())
            writing_score = float(self.writing_entry.get())

            # Validate input
            if not (0 <= reading_score <= 100 and 0 <= writing_score <= 100):
                messagebox.showerror("Invalid Input", "Scores must be between 0 and 100")
                return

            # Prepare input for prediction
            input_data = np.array([[reading_score, writing_score]])
            input_scaled = self.scaler.transform(input_data)

            # Predict
            predicted_score = self.model.predict(input_scaled)[0]

            # Update result label
            self.result_label.config(text=f"Predicted Math Score: {predicted_score:.2f}", 
                                     fg='#4CAF50')

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric scores")

def main():
    root = tk.Tk()
    app = MathScorePredictor(root)
    root.mainloop()

if __name__ == "__main__":
    main()
