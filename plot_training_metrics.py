

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from config import ImprovedConfig
import json
import re
from datetime import datetime

def extract_training_history_from_checkpoint(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        history_data = {}
        
        if 'detailed_history' in checkpoint:
            detailed = checkpoint['detailed_history']
            history_data['iterations'] = detailed.get('iterations', [])
            history_data['train_losses'] = detailed.get('train_losses', [])
            history_data['val_losses'] = detailed.get('val_losses', [])
            history_data['test_losses'] = detailed.get('test_losses', [])
            history_data['learning_rates'] = detailed.get('learning_rates', [])
            history_data['bleu_scores'] = detailed.get('bleu_scores', [])
            history_data['timestamps'] = detailed.get('timestamps', [])
        
        if 'translation_history' in checkpoint:
            history_data['translation_history'] = checkpoint['translation_history']
        
        if 'test_evaluation_history' in checkpoint:
            history_data['test_evaluation_history'] = checkpoint['test_evaluation_history']
        
        return history_data
    
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def parse_history_file(history_file_path):
    history_data = {
        'iterations': [],
        'train_losses': [],
        'val_losses': [],
        'test_losses': [],
        'learning_rates': [],
        'bleu_scores': [],
        'timestamps': []
    }
    
    try:
        with open(history_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        in_iteration_section = False
        
        for line in lines:
            line = line.strip()
            
            if 'ITERATION HISTORY' in line:
                in_iteration_section = True
                continue
            
            if in_iteration_section and line and not line.startswith('-') and not line.startswith('Iteration'):
                if 'TEST DATASET' in line:
                    break
                
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        iteration = int(parts[0])
                        train_loss = float(parts[1])
                        val_loss = float(parts[2])
                        test_loss = float(parts[3])
                        lr = float(parts[4])
                        bleu = float(parts[5])
                        timestamp = ' '.join(parts[6:]) if len(parts) > 6 else ''
                        
                        history_data['iterations'].append(iteration)
                        history_data['train_losses'].append(train_loss)
                        history_data['val_losses'].append(val_loss)
                        history_data['test_losses'].append(test_loss)
                        history_data['learning_rates'].append(lr)
                        history_data['bleu_scores'].append(bleu)
                        history_data['timestamps'].append(timestamp)
                    except ValueError:
                        continue
        
        return history_data if history_data['iterations'] else None
    
    except Exception as e:
        print(f"Error parsing history file: {e}")
        return None

def create_training_plots(history_data, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    
    if not history_data or not history_data.get('iterations'):
        print("No training history data available for plotting")
        return
    
    iterations = history_data['iterations']
    train_losses = history_data['train_losses']
    val_losses = history_data['val_losses']
    test_losses = history_data['test_losses']
    learning_rates = history_data['learning_rates']
    bleu_scores = history_data['bleu_scores']
    
    plt.style.use('default')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Metrics Over Time', fontsize=16, fontweight='bold')
    
    ax1 = axes[0, 0]
    ax1.plot(iterations, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(iterations, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.plot(iterations, test_losses, 'g-', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training, Validation & Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2 = axes[0, 1]
    ax2.plot(iterations, bleu_scores, 'purple', linewidth=2, marker='o', markersize=3)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('BLEU Score')
    ax2.set_title('BLEU Score Progress')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(bleu_scores) * 1.1 if bleu_scores else 1)
    
    ax3 = axes[1, 0]
    ax3.plot(iterations, learning_rates, 'orange', linewidth=2)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    ax4 = axes[1, 1]
    if len(train_losses) > 1 and len(val_losses) > 1:
        val_train_ratio = [v/t if t > 0 else 1 for v, t in zip(val_losses, train_losses)]
        ax4.plot(iterations, val_train_ratio, 'red', linewidth=2)
        ax4.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect Fit')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Val Loss / Train Loss')
        ax4.set_title('Overfitting Monitor')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Insufficient data\nfor overfitting analysis', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Overfitting Monitor')
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training metrics plot saved: {plot_path}")
    
    if len(iterations) > 10:
        fig2, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        window_size = max(1, len(iterations) // 20)
        
        def moving_average(data, window):
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        if len(train_losses) >= window_size:
            train_smooth = moving_average(train_losses, window_size)
            val_smooth = moving_average(val_losses, window_size)
            test_smooth = moving_average(test_losses, window_size)
            iter_smooth = iterations[window_size-1:]
            
            ax.plot(iter_smooth, train_smooth, 'b-', label=f'Train Loss (MA-{window_size})', linewidth=2)
            ax.plot(iter_smooth, val_smooth, 'r-', label=f'Val Loss (MA-{window_size})', linewidth=2)
            ax.plot(iter_smooth, test_smooth, 'g-', label=f'Test Loss (MA-{window_size})', linewidth=2)
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            ax.set_title('Smoothed Loss Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
            plt.tight_layout()
            smooth_plot_path = os.path.join(save_dir, 'smoothed_loss_curves.png')
            plt.savefig(smooth_plot_path, dpi=300, bbox_inches='tight')
            print(f"Smoothed loss curves saved: {smooth_plot_path}")
    
    plt.show()

def create_summary_report(history_data, save_dir="plots"):
    if not history_data or not history_data.get('iterations'):
        print("No training history data available for summary")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    iterations = history_data['iterations']
    train_losses = history_data['train_losses']
    val_losses = history_data['val_losses']
    test_losses = history_data['test_losses']
    bleu_scores = history_data['bleu_scores']
    
    report_path = os.path.join(save_dir, 'training_summary.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("TRAINING SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total Iterations: {len(iterations)}\n")
        f.write(f"Iteration Range: {min(iterations)} - {max(iterations)}\n\n")
        
        f.write("LOSS METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Initial Train Loss: {train_losses[0]:.4f}\n")
        f.write(f"Final Train Loss: {train_losses[-1]:.4f}\n")
        f.write(f"Best Train Loss: {min(train_losses):.4f}\n")
        f.write(f"Train Loss Reduction: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%\n\n")
        
        f.write(f"Initial Val Loss: {val_losses[0]:.4f}\n")
        f.write(f"Final Val Loss: {val_losses[-1]:.4f}\n")
        f.write(f"Best Val Loss: {min(val_losses):.4f}\n")
        f.write(f"Val Loss Reduction: {((val_losses[0] - val_losses[-1]) / val_losses[0] * 100):.1f}%\n\n")
        
        f.write(f"Initial Test Loss: {test_losses[0]:.4f}\n")
        f.write(f"Final Test Loss: {test_losses[-1]:.4f}\n")
        f.write(f"Best Test Loss: {min(test_losses):.4f}\n")
        f.write(f"Test Loss Reduction: {((test_losses[0] - test_losses[-1]) / test_losses[0] * 100):.1f}%\n\n")
        
        f.write("BLEU SCORE METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Initial BLEU: {bleu_scores[0]:.4f}\n")
        f.write(f"Final BLEU: {bleu_scores[-1]:.4f}\n")
        f.write(f"Best BLEU: {max(bleu_scores):.4f}\n")
        f.write(f"BLEU Improvement: {((bleu_scores[-1] - bleu_scores[0]) / max(bleu_scores[0], 0.001) * 100):.1f}%\n\n")
        
        final_val_train_ratio = val_losses[-1] / train_losses[-1] if train_losses[-1] > 0 else 1
        f.write("OVERFITTING ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Final Val/Train Loss Ratio: {final_val_train_ratio:.3f}\n")
        if final_val_train_ratio < 1.1:
            f.write("Status: Good generalization\n")
        elif final_val_train_ratio < 1.3:
            f.write("Status: Slight overfitting\n")
        else:
            f.write("Status: Significant overfitting\n")
    
    print(f"Training summary report saved: {report_path}")

def main():
    print("Training Metrics Plotter")
    print("=" * 40)
    
    history_data = None
    
    checkpoint_path = "best_translation_model.pt"
    if os.path.exists(checkpoint_path):
        print(f"Attempting to extract history from checkpoint: {checkpoint_path}")
        history_data = extract_training_history_from_checkpoint(checkpoint_path)
        if history_data and history_data.get('iterations'):
            print(f"Found {len(history_data['iterations'])} training iterations in checkpoint")
    
    if not history_data or not history_data.get('iterations'):
        history_files = [f for f in os.listdir('.') if 'history' in f.lower() and f.endswith('.txt')]
        
        if history_files:
            print(f"Found history files: {history_files}")
            for history_file in history_files:
                print(f"Attempting to parse: {history_file}")
                parsed_data = parse_history_file(history_file)
                if parsed_data and parsed_data.get('iterations'):
                    history_data = parsed_data
                    print(f"Successfully parsed {len(history_data['iterations'])} iterations")
                    break
    
    if history_data and history_data.get('iterations'):
        print(f"\nCreating plots for {len(history_data['iterations'])} training iterations...")
        create_training_plots(history_data)
        create_summary_report(history_data)
        print("Plotting completed!")
    else:
        print("\nNo training history data found.")
        print("Training history files are created when you run the improved_kaggle_train.py script.")
        print("They will be saved in the model_save_path directory (default: ./improved_checkpoints/)")
        
        print("\nCreating sample plot structure...")
        sample_data = {
            'iterations': list(range(0, 1000, 100)),
            'train_losses': [2.5 * (0.95 ** (i/100)) + 0.1 for i in range(0, 1000, 100)],
            'val_losses': [2.7 * (0.94 ** (i/100)) + 0.15 for i in range(0, 1000, 100)],
            'test_losses': [2.6 * (0.93 ** (i/100)) + 0.12 for i in range(0, 1000, 100)],
            'learning_rates': [1e-4 * (0.98 ** (i/100)) for i in range(0, 1000, 100)],
            'bleu_scores': [0.1 + 0.4 * (1 - 0.95 ** (i/100)) for i in range(0, 1000, 100)]
        }
        
        create_training_plots(sample_data, save_dir="sample_plots")
        print("Sample plots created in 'sample_plots' directory to show expected output format.")

if __name__ == "__main__":
    main()
