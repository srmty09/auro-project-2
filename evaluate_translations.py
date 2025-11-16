

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import re

def calculate_bleu_score(reference, candidate, n=4):
    
    def get_ngrams(text, n):
        words = text.lower().split()
        return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    
    def calculate_precision(ref_ngrams, cand_ngrams):
        if not cand_ngrams:
            return 0.0
        
        ref_counts = Counter(ref_ngrams)
        cand_counts = Counter(cand_ngrams)
        
        clipped_counts = 0
        total_counts = 0
        
        for ngram in cand_counts:
            clipped_counts += min(cand_counts[ngram], ref_counts.get(ngram, 0))
            total_counts += cand_counts[ngram]
        
        return clipped_counts / total_counts if total_counts > 0 else 0.0
    

    precisions = []
    for i in range(1, n + 1):
        ref_ngrams = get_ngrams(reference, i)
        cand_ngrams = get_ngrams(candidate, i)
        precision = calculate_precision(ref_ngrams, cand_ngrams)
        precisions.append(precision)
    

    ref_len = len(reference.split())
    cand_len = len(candidate.split())
    
    if cand_len > ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - ref_len / cand_len) if cand_len > 0 else 0.0
    

    if all(p > 0 for p in precisions):
        bleu = bp * np.exp(np.mean([np.log(p) for p in precisions]))
    else:
        bleu = 0.0
    
    return bleu, precisions

def calculate_word_overlap(reference, candidate):
    ref_words = set(reference.lower().split())
    cand_words = set(candidate.lower().split())
    
    if not ref_words and not cand_words:
        return 1.0
    if not ref_words or not cand_words:
        return 0.0
    
    intersection = ref_words.intersection(cand_words)
    union = ref_words.union(cand_words)
    
    return len(intersection) / len(union)

def evaluate_model_translations():
    

    odia_sentences = [
        "ଏହି ଟେଲିଭିଜନ ପୁରସ୍କାର ସମାରୋହ 2021 କାର୍ଯ୍ୟକ୍ରମ ତୁଳନାରେ ପ୍ରାୟ 1.5 ନିୟୁତ ଦର୍ଶକଙ୍କୁ ହରାଇଥିଲା ।",
        "ପୂର୍ବରେ ଋଷିଆ ସେନା ବିରୋଧରେ ୟୁକ୍ରେନୀୟ ସେନା ପକ୍ଷରୁ ଜାରି ହୋଇଥିବା ପ୍ରତିଆକ୍ରମଣରେ ନୂଆ ସଫଳତା ମିଳିଛି ।",
        "ଏମଏଲବି ଖେଳାଳି ସଂଘ ଶେଷରେ ଏଏଫଏଲ-ସିଆଇଓର ସଦସ୍ୟ ହେବ ଏବଂ ବିଭିନ୍ନ ଶିଳ୍ପର ଅନ୍ୟ ୫୭ଟି ସଂଘ ସହିତ ଯୋଡ଼ି ହେବ ।"
    ]
    

    model_translations = [
        "The telescope created nearly 1.5 million viewers in the 2021 program.",
        "There's a new success in the Russia military against the Russian force, said the U.S. government.",
        "The MLB Players Association will be a member of the AFL-CIO and another 75 other unions with other projects."
    ]
    

    reference_translations = [
        "The television awards ceremony lost roughly 1.5 million viewers compared to its 2021 program.",
        "Ukrainian forces are claiming new success in their counteroffensive against Russian forces in the east.",
        "The MLB Players Association will finally be a member of the AFL-CIO, affiliating with 57 other unions across industries."
    ]
    

    results = []
    for i, (odia, model, reference) in enumerate(zip(odia_sentences, model_translations, reference_translations)):
        bleu_score, precisions = calculate_bleu_score(reference, model)
        word_overlap = calculate_word_overlap(reference, model)
        
        result = {
            'sentence_id': i + 1,
            'odia': odia,
            'model_translation': model,
            'reference_translation': reference,
            'bleu_score': bleu_score,
            'precisions': precisions,
            'word_overlap': word_overlap
        }
        results.append(result)
    
    return results

def create_evaluation_plots(results, save_dir="evaluation_plots"):
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    sentence_ids = [r['sentence_id'] for r in results]
    bleu_scores = [r['bleu_score'] for r in results]
    word_overlaps = [r['word_overlap'] for r in results]
    

    precisions_1gram = [r['precisions'][0] for r in results]
    precisions_2gram = [r['precisions'][1] for r in results]
    precisions_3gram = [r['precisions'][2] for r in results]
    precisions_4gram = [r['precisions'][3] for r in results]
    
    plt.style.use('default')
    

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Translation Quality Evaluation', fontsize=16, fontweight='bold')
    

    ax1 = axes[0, 0]
    bars1 = ax1.bar(sentence_ids, bleu_scores, color='skyblue', alpha=0.7, edgecolor='navy')
    ax1.set_xlabel('Sentence ID')
    ax1.set_ylabel('BLEU Score')
    ax1.set_title('BLEU Scores by Sentence')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    

    for bar, score in zip(bars1, bleu_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    

    ax2 = axes[0, 1]
    bars2 = ax2.bar(sentence_ids, word_overlaps, color='lightgreen', alpha=0.7, edgecolor='darkgreen')
    ax2.set_xlabel('Sentence ID')
    ax2.set_ylabel('Word Overlap Score')
    ax2.set_title('Word Overlap Similarity')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    

    for bar, score in zip(bars2, word_overlaps):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    

    ax3 = axes[1, 0]
    x = np.arange(len(sentence_ids))
    width = 0.2
    
    ax3.bar(x - 1.5*width, precisions_1gram, width, label='1-gram', alpha=0.8)
    ax3.bar(x - 0.5*width, precisions_2gram, width, label='2-gram', alpha=0.8)
    ax3.bar(x + 0.5*width, precisions_3gram, width, label='3-gram', alpha=0.8)
    ax3.bar(x + 1.5*width, precisions_4gram, width, label='4-gram', alpha=0.8)
    
    ax3.set_xlabel('Sentence ID')
    ax3.set_ylabel('Precision Score')
    ax3.set_title('N-gram Precision Scores')
    ax3.set_xticks(x)
    ax3.set_xticklabels(sentence_ids)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    

    ax4 = axes[1, 1]
    avg_bleu = np.mean(bleu_scores)
    avg_overlap = np.mean(word_overlaps)
    
    metrics = ['Average BLEU', 'Average Word Overlap']
    values = [avg_bleu, avg_overlap]
    colors = ['skyblue', 'lightgreen']
    
    bars4 = ax4.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Score')
    ax4.set_title('Overall Performance Summary')
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)
    

    for bar, value in zip(bars4, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'translation_evaluation.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Evaluation plot saved: {plot_path}")
    

    fig2, ax = plt.subplots(1, 1, figsize=(14, 8))
    

    combined_scores = [0.7 * bleu + 0.3 * overlap for bleu, overlap in zip(bleu_scores, word_overlaps)]
    
    x = np.arange(len(sentence_ids))
    width = 0.25
    
    bars1 = ax.bar(x - width, bleu_scores, width, label='BLEU Score', alpha=0.8, color='skyblue')
    bars2 = ax.bar(x, word_overlaps, width, label='Word Overlap', alpha=0.8, color='lightgreen')
    bars3 = ax.bar(x + width, combined_scores, width, label='Combined Score', alpha=0.8, color='orange')
    
    ax.set_xlabel('Sentence ID')
    ax.set_ylabel('Score')
    ax.set_title('Detailed Translation Quality Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(sentence_ids)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    detailed_plot_path = os.path.join(save_dir, 'detailed_comparison.png')
    plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
    print(f"Detailed comparison plot saved: {detailed_plot_path}")
    
    plt.show()

def create_evaluation_report(results, save_dir="evaluation_plots"):
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    report_path = os.path.join(save_dir, 'evaluation_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("TRANSLATION QUALITY EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        

        bleu_scores = [r['bleu_score'] for r in results]
        word_overlaps = [r['word_overlap'] for r in results]
        
        f.write("OVERALL PERFORMANCE:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Average BLEU Score: {np.mean(bleu_scores):.4f}\n")
        f.write(f"Average Word Overlap: {np.mean(word_overlaps):.4f}\n")
        f.write(f"Best BLEU Score: {max(bleu_scores):.4f} (Sentence {bleu_scores.index(max(bleu_scores)) + 1})\n")
        f.write(f"Worst BLEU Score: {min(bleu_scores):.4f} (Sentence {bleu_scores.index(min(bleu_scores)) + 1})\n\n")
        

        f.write("DETAILED SENTENCE ANALYSIS:\n")
        f.write("=" * 60 + "\n\n")
        
        for result in results:
            f.write(f"SENTENCE {result['sentence_id']}:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Odia: {result['odia']}\n\n")
            f.write(f"Model Translation:\n{result['model_translation']}\n\n")
            f.write(f"Reference Translation:\n{result['reference_translation']}\n\n")
            f.write(f"BLEU Score: {result['bleu_score']:.4f}\n")
            f.write(f"Word Overlap: {result['word_overlap']:.4f}\n")
            f.write(f"1-gram Precision: {result['precisions'][0]:.4f}\n")
            f.write(f"2-gram Precision: {result['precisions'][1]:.4f}\n")
            f.write(f"3-gram Precision: {result['precisions'][2]:.4f}\n")
            f.write(f"4-gram Precision: {result['precisions'][3]:.4f}\n")
            f.write("\n" + "=" * 60 + "\n\n")
        

        f.write("ANALYSIS AND RECOMMENDATIONS:\n")
        f.write("-" * 40 + "\n")
        
        avg_bleu = np.mean(bleu_scores)
        if avg_bleu > 0.5:
            f.write("✓ Good overall translation quality\n")
        elif avg_bleu > 0.3:
            f.write("~ Moderate translation quality - room for improvement\n")
        else:
            f.write("✗ Poor translation quality - significant improvement needed\n")
        
        f.write(f"\nRecommendations:\n")
        if avg_bleu < 0.3:
            f.write("- Increase training data quantity and quality\n")
            f.write("- Train for more epochs\n")
            f.write("- Consider using a larger model (mT5-base)\n")
        elif avg_bleu < 0.5:
            f.write("- Fine-tune hyperparameters\n")
            f.write("- Add more domain-specific training data\n")
            f.write("- Implement better preprocessing\n")
        else:
            f.write("- Model performance is good\n")
            f.write("- Consider domain adaptation for specific use cases\n")
    
    print(f"Evaluation report saved: {report_path}")

def main():
    print("Translation Quality Evaluation")
    print("=" * 50)
    

    results = evaluate_model_translations()
    

    print(f"\nEvaluated {len(results)} translations:")
    for result in results:
        print(f"Sentence {result['sentence_id']}: BLEU={result['bleu_score']:.3f}, Overlap={result['word_overlap']:.3f}")
    
    avg_bleu = np.mean([r['bleu_score'] for r in results])
    avg_overlap = np.mean([r['word_overlap'] for r in results])
    
    print(f"\nOverall Performance:")
    print(f"Average BLEU Score: {avg_bleu:.3f}")
    print(f"Average Word Overlap: {avg_overlap:.3f}")
    

    create_evaluation_plots(results)
    create_evaluation_report(results)
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()
