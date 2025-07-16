import tkinter as tk
from tkinter import font
import random
import numpy as np


machines = [
    {"name": "Machine 1", "mean": 1, "std": 1.0, "profit": 0.0, "pulls": 0},
    {"name": "Machine 2", "mean": 0, "std": 1.0, "profit": 0.0, "pulls": 0},
    {"name": "Machine 3", "mean": -0.5, "std": 1.0, "profit": 0.0, "pulls": 0},
]

machine_labels = []  
machine_buttons = []

MAX_PULLS = 50 

def update_machine_label(index):
    
    machine = machines[index]
    avg = machine["profit"] / machine["pulls"] if machine["pulls"] > 0 else 0.0
    text = (f"{machine['name']}\n"
            f"Profit: £{machine['profit']:.2f}\n"
            f"Pulls: {machine['pulls']}\n"
            f"Average: {avg:.2f}")
    machine_labels[index].config(text=text)
    machine_labels[index].update_idletasks()

def disable_buttons():
    
    for btn in machine_buttons:
        btn.config(state=tk.DISABLED)

def compute_and_display_regrets():
    
    highest_avg = 0.0
    best_machine_index = None

    for i, m in enumerate(machines):
        if m["pulls"] > 0:
            avg_reward = m["profit"] / m["pulls"]
            if avg_reward > highest_avg:
                highest_avg = avg_reward
                best_machine_index = i

    total_profit = sum(m["profit"] for m in machines)
    total_pulls = sum(m["pulls"] for m in machines) 
    actual_avg = total_profit / total_pulls if total_pulls > 0 else 0.0

    total_regret = highest_avg * MAX_PULLS - total_profit
    average_regret = highest_avg - actual_avg
    if best_machine_index is not None:
        action_regret = MAX_PULLS - machines[best_machine_index]["pulls"]
    else:
        action_regret = MAX_PULLS

    text = (f"TOTAL STATS\n"
            f"Total Profit: £{total_profit:.2f}\n"
            f"Total Pulls: {total_pulls}\n"
            f"Overall Average: {actual_avg:.2f}\n\n"
            f"--- REGRETS ---\n"
            f"Highest Average: {highest_avg:.2f}\n"
            f"Total Regret: £{total_regret:.2f}\n"
            f"Average Regret: {average_regret:.2f}\n"
            f"Action Regret: {action_regret}")
    total_label.config(text=text)
    total_label.update_idletasks()

def update_total_stats():
    
    total_profit = sum(m["profit"] for m in machines)
    total_pulls = sum(m["pulls"] for m in machines)
    overall_avg = total_profit / total_pulls if total_pulls > 0 else 0.0

    text = (f"TOTAL STATS\n"
            f"Total Profit: £{total_profit:.2f}\n"
            f"Total Pulls: {total_pulls}\n"
            f"Overall Average: {overall_avg:.2f}")
    total_label.config(text=text)
    total_label.update_idletasks()

def display_reward(machine_index, reward):
   
    output_canvas.delete("all")
    output_canvas.update_idletasks()
    panel_width, panel_height = 400, 150
    width = output_canvas.winfo_width()
    height = output_canvas.winfo_height()
    x0 = (width - panel_width) // 2
    y0 = (height - panel_height) // 2
    x1, y1 = x0 + panel_width, y0 + panel_height

    output_canvas.create_rectangle(x0, y0, x1, y1, fill="black", outline="gold", width=4)
    text = f"{machines[machine_index]['name']}\nReward: £{reward:.2f}"
    output_canvas.create_text((x0+x1)//2, (y0+y1)//2,
                              text=text,
                              fill="lime",
                              font=font.Font(size=24, weight="bold"),
                              justify="center")
    output_canvas.update()

def pull_machine(index):

    total_pulls_done = sum(m["pulls"] for m in machines)
    if total_pulls_done >= MAX_PULLS:
        return

    machine = machines[index]
    reward = np.random.normal(machine["mean"], machine["std"])
    machine["profit"] += reward
    machine["pulls"] += 1

    update_machine_label(index)
    total_pulls_done = sum(m["pulls"] for m in machines)
    if total_pulls_done < MAX_PULLS:
        update_total_stats()
    else:
        compute_and_display_regrets()
        disable_buttons()

    display_reward(index, reward)


root = tk.Tk()
root.title("Multi-Armed Bandit Simulator (50 Pulls)")
root.configure(bg="gray20")

control_frame = tk.Frame(root, bg="gray20")
control_frame.pack(padx=10, pady=10)


machines_frame = tk.Frame(control_frame, bg="gray20")
machines_frame.grid(row=0, column=0, padx=10, pady=5)

stats_frame = tk.Frame(control_frame, bg="gray20")
stats_frame.grid(row=0, column=1, padx=10, pady=5)


machine_font = ("Courier", 12)
for i, machine in enumerate(machines):
    panel = tk.Frame(machines_frame, bg="gray30", bd=2, relief="groove")
    panel.grid(row=0, column=i, padx=10, pady=5)
    
    label = tk.Label(panel,
                     text=f"{machine['name']}\nProfit: £0.00\nPulls: 0\nAverage: 0.00",
                     width=20, height=5, bg="black", fg="lime",
                     font=machine_font)
    label.pack(padx=5, pady=5)
    machine_labels.append(label)
    
    btn = tk.Button(panel, text=f"Pull {machine['name']}",
                    command=lambda i=i: pull_machine(i),
                    bg="gold", fg="black", font=("Helvetica", 10, "bold"))
    btn.pack(padx=5, pady=5)
    machine_buttons.append(btn)


total_label = tk.Label(stats_frame,
                       text="TOTAL STATS\nTotal Profit: £0.00\nTotal Pulls: 0\nOverall Average: 0.00",
                       bg="black", fg="cyan", width=28, height=9,
                       font=("Courier", 12), justify="left")
total_label.pack(padx=10, pady=10)


output_frame = tk.Frame(root, bg="gray20")
output_frame.pack(padx=10, pady=10, fill="both", expand=True)

output_canvas = tk.Canvas(output_frame, bg="gray10", height=200)
output_canvas.pack(fill="both", expand=True, padx=20, pady=20)


update_total_stats()

root.mainloop()
