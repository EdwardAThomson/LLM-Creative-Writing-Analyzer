#!/usr/bin/env python3
# llm_tester_ui.py

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import queue
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer


from utils import (
    load_parameters,
    create_prompt,
    ensure_output_dir
)
from ai_helper import send_prompt
from llm_creative_tester import run_tests, DEFAULT_OUTPUT_DIR, DEFAULT_REPEATS, DEFAULT_WORD_COUNT, DEFAULT_PAUSE

# List of available models from ai_helper.py
AVAILABLE_MODELS = [
    "gpt-4o",
    "o1",
    "o1-mini",
    "gemini-1.5-pro",
    "gemini-2.0-pro-exp-02-05",
    "gemini-2.5-pro-exp-03-25",
    "claude-3-opus",
    "claude-3-5-sonnet",
    "claude-3-7-sonnet"
]

class RedirectText:
    """Class to redirect stdout to a tkinter Text widget"""
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.queue = queue.Queue()
        self.updating = True
        threading.Thread(target=self.update_text_widget, daemon=True).start()

    def write(self, string):
        self.queue.put(string)

    def flush(self):
        pass

    def update_text_widget(self):
        while self.updating:
            try:
                while True:
                    string = self.queue.get_nowait()
                    self.text_widget.configure(state="normal")
                    self.text_widget.insert(tk.END, string)
                    self.text_widget.see(tk.END)
                    self.text_widget.configure(state="disabled")
                    self.queue.task_done()
            except queue.Empty:
                pass
            self.text_widget.update()
            from time import sleep
            sleep(0.1)

    def close(self):
        self.updating = False

class LLMTesterUI:
    """Main UI class for the LLM Creative Writing Tester"""
    def __init__(self, root):
        self.root = root
        self.root.title("LLM Creative Writing Tester")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # Initialize variables
        self.params_file_path = "parameters.txt"
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.setup_tab = ttk.Frame(self.notebook)
        self.output_tab = ttk.Frame(self.notebook)
        self.parameters_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.setup_tab, text="Test Setup")
        self.notebook.add(self.parameters_tab, text="Parameters")
        self.notebook.add(self.output_tab, text="Output")
        
        # Create Setup Tab content
        self.create_setup_tab()
        
        # Create Parameters Tab content
        self.create_parameters_tab()
        
        # Create Output Tab content
        self.create_output_tab()
        
        # Load parameters
        self.load_parameters_text()
        
        # Add status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def create_setup_tab(self):
        """Create the Test Setup tab content"""
        frame = ttk.Frame(self.setup_tab, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Models selection
        ttk.Label(frame, text="Select Models to Test:", font=("", 10, "bold")).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        models_frame = ttk.Frame(frame)
        models_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        # Create checkboxes for each model
        self.model_vars = {}
        for i, model in enumerate(AVAILABLE_MODELS):
            var = tk.BooleanVar(value=model in ["gpt-4o", "gemini-1.5-pro"])
            self.model_vars[model] = var
            ttk.Checkbutton(models_frame, text=model, variable=var).grid(row=i//2, column=i%2, sticky=tk.W, padx=(0, 10))
        
        # Test settings
        settings_frame = ttk.LabelFrame(frame, text="Test Settings", padding="10")
        settings_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Repeats
        ttk.Label(settings_frame, text="Repeats:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.repeats_var = tk.IntVar(value=DEFAULT_REPEATS)
        repeats_spinner = ttk.Spinbox(settings_frame, from_=1, to=10, textvariable=self.repeats_var, width=5)
        repeats_spinner.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Word count
        ttk.Label(settings_frame, text="Target Word Count:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.word_count_var = tk.IntVar(value=DEFAULT_WORD_COUNT)
        word_count_spinner = ttk.Spinbox(settings_frame, from_=100, to=3000, increment=100, textvariable=self.word_count_var, width=5)
        word_count_spinner.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Pause between API calls
        ttk.Label(settings_frame, text="Pause Between Calls (s):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.pause_var = tk.DoubleVar(value=DEFAULT_PAUSE)
        pause_spinner = ttk.Spinbox(settings_frame, from_=0.1, to=10, increment=0.1, textvariable=self.pause_var, width=5)
        pause_spinner.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Output directory
        ttk.Label(settings_frame, text="Output Directory:").grid(row=3, column=0, sticky=tk.W, pady=5)
        output_frame = ttk.Frame(settings_frame)
        output_frame.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        self.output_dir_var = tk.StringVar(value=DEFAULT_OUTPUT_DIR)
        output_entry = ttk.Entry(output_frame, textvariable=self.output_dir_var, width=30)
        output_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        def browse_output_dir():
            directory = filedialog.askdirectory(initialdir=self.output_dir_var.get())
            if directory:
                self.output_dir_var.set(directory)
        
        browse_button = ttk.Button(output_frame, text="Browse...", command=browse_output_dir)
        browse_button.pack(side=tk.LEFT)
        
        # Parameters file
        ttk.Label(settings_frame, text="Parameters File:").grid(row=4, column=0, sticky=tk.W, pady=5)
        params_frame = ttk.Frame(settings_frame)
        params_frame.grid(row=4, column=1, sticky=tk.W, pady=5)
        
        self.params_file_var = tk.StringVar(value=self.params_file_path)
        params_entry = ttk.Entry(params_frame, textvariable=self.params_file_var, width=30)
        params_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        def browse_params_file():
            file_path = filedialog.askopenfilename(
                initialdir=os.path.dirname(self.params_file_var.get()),
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if file_path:
                self.params_file_var.set(file_path)
                self.params_file_path = file_path
                self.load_parameters_text()
        
        browse_button = ttk.Button(params_frame, text="Browse...", command=browse_params_file)
        browse_button.pack(side=tk.LEFT)
        
        # Run button
        run_button = ttk.Button(frame, text="Run Tests", command=self.run_tests)
        run_button.grid(row=3, column=0, pady=10)
        
        # Help text
        help_text = ttk.Label(
            frame, 
            text="Select at least one model and configure your test settings, then click 'Run Tests'.\n"
                 "You can view and edit the parameters in the 'Parameters' tab. Output will be shown in the 'Output' tab.",
            wraplength=700, justify=tk.LEFT
        )
        help_text.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=10)
    
    def create_parameters_tab(self):
        """Create the Parameters tab content"""
        frame = ttk.Frame(self.parameters_tab, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Story Parameters", font=("", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        # Parameters text area
        self.parameters_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=25)
        self.parameters_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Buttons frame
        buttons_frame = ttk.Frame(frame)
        buttons_frame.pack(fill=tk.X)
        
        def save_parameters():
            try:
                with open(self.params_file_path, 'w') as f:
                    f.write(self.parameters_text.get(1.0, tk.END))
                self.status_var.set(f"Parameters saved to {self.params_file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save parameters: {str(e)}")
                self.status_var.set("Error saving parameters")
        
        save_button = ttk.Button(buttons_frame, text="Save Parameters", command=save_parameters)
        save_button.pack(side=tk.LEFT, padx=(0, 10))
        
        def refresh_parameters():
            self.load_parameters_text()
            self.status_var.set(f"Parameters reloaded from {self.params_file_path}")
        
        refresh_button = ttk.Button(buttons_frame, text="Reload from File", command=refresh_parameters)
        refresh_button.pack(side=tk.LEFT)
    
    def create_output_tab(self):
        """Create the Output tab content"""
        frame = ttk.Frame(self.output_tab, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Test Output", font=("", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        # Output text area
        self.output_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=25)
        self.output_text.pack(fill=tk.BOTH, expand=True)
        self.output_text.configure(state="disabled")
        
        # Buttons frame
        buttons_frame = ttk.Frame(frame)
        buttons_frame.pack(fill=tk.X, pady=(10, 0))
        
        def clear_output():
            self.output_text.configure(state="normal")
            self.output_text.delete(1.0, tk.END)
            self.output_text.configure(state="disabled")
            self.status_var.set("Output cleared")
        
        clear_button = ttk.Button(buttons_frame, text="Clear Output", command=clear_output)
        clear_button.pack(side=tk.LEFT, padx=(0, 10))
        
        def open_results_dir():
            output_dir = self.output_dir_var.get()
            if not os.path.exists(output_dir):
                messagebox.showinfo("Information", f"Output directory '{output_dir}' doesn't exist yet")
                return
            
            # Open file explorer to the output directory
            if sys.platform == 'win32':
                os.startfile(output_dir)
            elif sys.platform == 'darwin':  # macOS
                os.system(f'open "{output_dir}"')
            else:  # Linux
                os.system(f'xdg-open "{output_dir}"')
        
        open_dir_button = ttk.Button(buttons_frame, text="Open Results Directory", command=open_results_dir)
        open_dir_button.pack(side=tk.LEFT)
    
    def load_parameters_text(self):
        """Load parameters text from file"""
        try:
            with open(self.params_file_path, 'r') as f:
                params_text = f.read()
                
            if hasattr(self, 'parameters_text'):
                self.parameters_text.delete(1.0, tk.END)
                self.parameters_text.insert(tk.END, params_text)
        except Exception as e:
            if hasattr(self, 'parameters_text'):
                self.parameters_text.delete(1.0, tk.END)
                self.parameters_text.insert(tk.END, f"Error loading parameters: {str(e)}")
    
    def run_tests(self):
        """Run the LLM tests based on the UI configuration"""
        # Get selected models
        selected_models = [model for model, var in self.model_vars.items() if var.get()]
        
        if not selected_models:
            messagebox.showwarning("Warning", "Please select at least one model to test")
            return
        
        # Get other settings
        repeats = self.repeats_var.get()
        word_count = self.word_count_var.get()
        pause_seconds = self.pause_var.get()
        output_dir = self.output_dir_var.get()
        params_file = self.params_file_var.get()
        
        # Validate settings
        if repeats < 1:
            messagebox.showwarning("Warning", "Repeats must be at least 1")
            return
        
        if word_count < 100:
            messagebox.showwarning("Warning", "Word count must be at least 100")
            return
        
        if pause_seconds < 0:
            messagebox.showwarning("Warning", "Pause seconds must be 0 or greater")
            return
        
        # Load parameters
        try:
            parameters_text = load_parameters(params_file)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load parameters: {str(e)}")
            return
        
        # Prepare to run tests
        self.status_var.set(f"Running tests with {len(selected_models)} models, {repeats} repeats each...")
        self.notebook.select(self.output_tab)
        
        # Redirect stdout to output text
        self.output_text.configure(state="normal")
        self.output_text.delete(1.0, tk.END)
        self.output_text.configure(state="disabled")
        
        redirect = RedirectText(self.output_text)
        old_stdout = sys.stdout
        sys.stdout = redirect
        
        # Run tests in a separate thread to avoid freezing the UI
        def run_tests_thread():
            try:
                results, timestamp = run_tests(
                    models=selected_models,
                    parameters_text=parameters_text,
                    repeats=repeats,
                    word_count=word_count,
                    output_dir=output_dir,
                    pause_seconds=pause_seconds
                )
                
                # Show completion message in the UI thread
                self.root.after(0, lambda: self.status_var.set(f"Tests completed. Results saved to {output_dir}/"))
                
            except Exception as e:
                # Show error in the UI thread
                self.root.after(0, lambda: messagebox.showerror("Error", f"Error running tests: {str(e)}"))
                self.root.after(0, lambda: self.status_var.set("Error running tests"))
            finally:
                # Restore stdout in the UI thread
                self.root.after(0, lambda: setattr(sys, 'stdout', old_stdout))
                redirect.close()
        
        threading.Thread(target=run_tests_thread, daemon=True).start()

def calculate_semantic_similarity(text1, text2):
    """Calculate semantic similarity between texts using embeddings."""
    
    
    # Load model (first time will download it)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create embeddings
    embedding1 = model.encode(text1)
    embedding2 = model.encode(text2)
    
    # Calculate cosine similarity
    
    return cosine_similarity(
        embedding1.reshape(1, -1),
        embedding2.reshape(1, -1)
    )[0][0]

def main():
    root = tk.Tk()
    app = LLMTesterUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 