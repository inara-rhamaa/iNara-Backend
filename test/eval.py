import pandas as pd
import numpy as np
import os
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
from difflib import SequenceMatcher
import json
from datetime import datetime

# Simple plotting without seaborn dependency
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualizations")

class AIConsistencyAnalyzer:
    """
    Lightweight analyzer untuk mengevaluasi konsistensi jawaban AI (RAG vs Original AI)
    dari output CSV yang dihasilkan oleh tester RAG vs OG.
    Output: CSV files dan PNG plots saja (cocok untuk VM environment)
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize analyzer dengan path ke CSV hasil testing.
        
        Args:
            csv_path: Path ke file CSV hasil testing
        """
        self.csv_path = csv_path
        self.df = self._load_data()
        self.analysis_results = {}
        
    def _load_data(self) -> pd.DataFrame:
        """Load dan validasi data dari CSV dengan error handling yang lebih baik."""
        try:
            # Check if file exists
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(f"File tidak ditemukan: {self.csv_path}")
            
            # Check if file is empty
            if os.path.getsize(self.csv_path) == 0:
                raise ValueError(f"File kosong: {self.csv_path}")
            
            # Try to read CSV
            df = pd.read_csv(self.csv_path)
            
            if df.empty:
                raise ValueError(f"CSV tidak memiliki data: {self.csv_path}")
            
            # Check required columns - lebih fleksibel
            required_cols = ['pertanyaan']  # Minimal requirement
            optional_cols = ['ragAI', 'ogAI', 'rag_benar', 'og_benar', 
                           'rag_score', 'og_score', 'rag_verdict', 'og_verdict']
            
            missing_required = [col for col in required_cols if col not in df.columns]
            if missing_required:
                raise ValueError(f"Kolom wajib yang hilang: {missing_required}")
            
            # Check which optional columns are available
            available_cols = [col for col in optional_cols if col in df.columns]
            print(f"Kolom tersedia: {list(df.columns)}")
            print(f"Kolom opsional ditemukan: {available_cols}")
            
            # Convert numeric columns if they exist
            for col in ['rag_score', 'og_score']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            print(f"Data berhasil dimuat: {len(df)} baris, {len(df.columns)} kolom")
            return df
            
        except Exception as e:
            raise ValueError(f"Error memuat data dari {self.csv_path}: {e}")
    
    def calculate_basic_metrics(self) -> Dict:
        """Hitung metrik dasar performa AI."""
        total_questions = len(self.df)
        metrics = {'total_questions': total_questions}
        
        # Check availability and calculate metrics
        if 'rag_verdict' in self.df.columns and 'og_verdict' in self.df.columns:
            rag_correct = len(self.df[self.df['rag_verdict'] == 'BENAR'])
            og_correct = len(self.df[self.df['og_verdict'] == 'BENAR'])
            metrics['rag_accuracy_verdict'] = rag_correct / total_questions * 100
            metrics['og_accuracy_verdict'] = og_correct / total_questions * 100
        
        if 'rag_benar' in self.df.columns and 'og_benar' in self.df.columns:
            rag_true_count = len(self.df[self.df['rag_benar'] == 'TRUE'])
            og_true_count = len(self.df[self.df['og_benar'] == 'TRUE'])
            metrics['rag_accuracy_boolean'] = rag_true_count / total_questions * 100
            metrics['og_accuracy_boolean'] = og_true_count / total_questions * 100
        
        # Score statistics
        if 'rag_score' in self.df.columns:
            rag_scores = self.df['rag_score'].dropna()
            if len(rag_scores) > 0:
                metrics.update({
                    'rag_avg_score': rag_scores.mean(),
                    'rag_median_score': rag_scores.median(),
                    'rag_std_score': rag_scores.std()
                })
        
        if 'og_score' in self.df.columns:
            og_scores = self.df['og_score'].dropna()
            if len(og_scores) > 0:
                metrics.update({
                    'og_avg_score': og_scores.mean(),
                    'og_median_score': og_scores.median(),
                    'og_std_score': og_scores.std()
                })
        
        self.analysis_results['basic_metrics'] = metrics
        return metrics
    
    def analyze_consistency(self) -> Dict:
        """Analisis konsistensi antara RAG AI dan Original AI."""
        consistency_metrics = {}
        
        # 1. Agreement dalam verdict
        if 'rag_verdict' in self.df.columns and 'og_verdict' in self.df.columns:
            verdict_agreement = (self.df['rag_verdict'] == self.df['og_verdict']).sum()
            consistency_metrics['verdict_agreement_rate'] = verdict_agreement / len(self.df) * 100
        
        # 2. Agreement dalam boolean result
        if 'rag_benar' in self.df.columns and 'og_benar' in self.df.columns:
            bool_agreement = (self.df['rag_benar'] == self.df['og_benar']).sum()
            consistency_metrics['bool_agreement_rate'] = bool_agreement / len(self.df) * 100
        
        # 3. Score correlation dan difference
        if 'rag_score' in self.df.columns and 'og_score' in self.df.columns:
            valid_scores = self.df.dropna(subset=['rag_score', 'og_score'])
            if len(valid_scores) > 1:
                score_correlation = valid_scores['rag_score'].corr(valid_scores['og_score'])
                score_diff = abs(valid_scores['rag_score'] - valid_scores['og_score'])
                consistency_metrics.update({
                    'score_correlation': score_correlation,
                    'avg_score_difference': score_diff.mean(),
                    'max_score_difference': score_diff.max()
                })
        
        # 4. Win-Loss analysis
        if 'rag_benar' in self.df.columns and 'og_benar' in self.df.columns:
            rag_wins = len(self.df[(self.df['rag_benar'] == 'TRUE') & (self.df['og_benar'] == 'FALSE')])
            og_wins = len(self.df[(self.df['og_benar'] == 'TRUE') & (self.df['rag_benar'] == 'FALSE')])
            consistency_metrics.update({'rag_wins': rag_wins, 'og_wins': og_wins})
        
        # 5. Semantic similarity
        if 'ragAI' in self.df.columns and 'ogAI' in self.df.columns:
            semantic_similarities = []
            for _, row in self.df.iterrows():
                rag_ans = str(row['ragAI']).lower().strip()
                og_ans = str(row['ogAI']).lower().strip()
                similarity = SequenceMatcher(None, rag_ans, og_ans).ratio()
                semantic_similarities.append(similarity)
            
            consistency_metrics['avg_semantic_similarity'] = np.mean(semantic_similarities)
            consistency_metrics['semantic_similarities'] = semantic_similarities
        
        self.analysis_results['consistency'] = consistency_metrics
        return consistency_metrics
    
    def identify_problematic_questions(self, min_score_diff: float = 0.3) -> pd.DataFrame:
        """Identifikasi pertanyaan yang menghasilkan jawaban tidak konsisten."""
        
        mask = pd.Series(False, index=self.df.index)
        
        # Filter berdasarkan perbedaan score
        if all(col in self.df.columns for col in ['rag_score', 'og_score']):
            score_diff = abs(self.df['rag_score'] - self.df['og_score'])
            high_diff_mask = score_diff >= min_score_diff
            mask |= high_diff_mask
        
        # Filter berdasarkan disagreement dalam verdict
        if all(col in self.df.columns for col in ['rag_verdict', 'og_verdict']):
            verdict_disagree_mask = self.df['rag_verdict'] != self.df['og_verdict']
            mask |= verdict_disagree_mask
        
        # Filter berdasarkan disagreement dalam boolean result
        if all(col in self.df.columns for col in ['rag_benar', 'og_benar']):
            bool_disagree_mask = self.df['rag_benar'] != self.df['og_benar']
            mask |= bool_disagree_mask
        
        problematic_df = self.df[mask].copy()
        
        # Add score difference if available
        if all(col in self.df.columns for col in ['rag_score', 'og_score']):
            problematic_df['score_difference'] = abs(self.df['rag_score'] - self.df['og_score'])[mask]
            problematic_df = problematic_df.sort_values('score_difference', ascending=False)
        
        return problematic_df
    
    def generate_simple_plots(self, output_dir: str = "plots"):
        """Generate plots sederhana tanpa seaborn."""
        if not HAS_MATPLOTLIB:
            print("Matplotlib tidak tersedia, skip visualisasi")
            return
        
        Path(output_dir).mkdir(exist_ok=True)
        plt.style.use('default')
        
        # Plot 1: Score Distribution (jika ada)
        if all(col in self.df.columns for col in ['rag_score', 'og_score']):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            rag_scores = self.df['rag_score'].dropna()
            og_scores = self.df['og_score'].dropna()
            
            ax.hist(rag_scores, bins=20, alpha=0.7, label='RAG AI', color='blue', edgecolor='black')
            ax.hist(og_scores, bins=20, alpha=0.7, label='Original AI', color='red', edgecolor='black')
            ax.set_xlabel('Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Score Distribution Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/score_distribution.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Score distribution plot saved to {output_dir}/score_distribution.png")
        
        # Plot 2: Score Correlation (jika ada)
        if all(col in self.df.columns for col in ['rag_score', 'og_score']):
            valid_data = self.df.dropna(subset=['rag_score', 'og_score'])
            if len(valid_data) > 0:
                fig, ax = plt.subplots(figsize=(8, 8))
                
                ax.scatter(valid_data['rag_score'], valid_data['og_score'], alpha=0.6, color='green')
                ax.plot([0, 1], [0, 1], 'r--', label='Perfect Agreement', linewidth=2)
                ax.set_xlabel('RAG AI Score')
                ax.set_ylabel('Original AI Score')
                ax.set_title('Score Correlation')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/score_correlation.png", dpi=150, bbox_inches='tight')
                plt.close()
                print(f"✓ Score correlation plot saved to {output_dir}/score_correlation.png")
        
        # Plot 3: Performance Comparison Bar Chart
        metrics = self.analysis_results.get('basic_metrics', {})
        
        categories = []
        rag_values = []
        og_values = []
        
        if 'rag_accuracy_verdict' in metrics and 'og_accuracy_verdict' in metrics:
            categories.append('Accuracy (Verdict)')
            rag_values.append(metrics['rag_accuracy_verdict'])
            og_values.append(metrics['og_accuracy_verdict'])
        
        if 'rag_accuracy_boolean' in metrics and 'og_accuracy_boolean' in metrics:
            categories.append('Accuracy (Boolean)')
            rag_values.append(metrics['rag_accuracy_boolean'])
            og_values.append(metrics['og_accuracy_boolean'])
        
        if 'rag_avg_score' in metrics and 'og_avg_score' in metrics:
            categories.append('Avg Score')
            rag_values.append(metrics['rag_avg_score'] * 100)
            og_values.append(metrics['og_avg_score'] * 100)
        
        if categories:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.arange(len(categories))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, rag_values, width, label='RAG AI', color='blue', alpha=0.7)
            bars2 = ax.bar(x + width/2, og_values, width, label='Original AI', color='red', alpha=0.7)
            
            ax.set_ylabel('Percentage (%)')
            ax.set_title('AI Performance Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.1f}',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3),
                              textcoords="offset points",
                              ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/performance_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Performance comparison plot saved to {output_dir}/performance_comparison.png")
        
        # Plot 4: Semantic Similarity (jika ada)
        consistency = self.analysis_results.get('consistency', {})
        if 'semantic_similarities' in consistency:
            similarities = consistency['semantic_similarities']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.hist(similarities, bins=30, alpha=0.7, color='green', edgecolor='black')
            ax.axvline(np.mean(similarities), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(similarities):.3f}')
            ax.set_xlabel('Semantic Similarity Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Semantic Similarity Between RAG and Original AI Answers')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/semantic_similarity.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Semantic similarity plot saved to {output_dir}/semantic_similarity.png")
    
    def save_results_to_csv(self, output_dir: str):
        """Save analysis results ke CSV files."""
        Path(output_dir).mkdir(exist_ok=True)
        
        # 1. Basic metrics
        basic_metrics = self.analysis_results.get('basic_metrics', {})
        if basic_metrics:
            metrics_df = pd.DataFrame([basic_metrics])
            metrics_path = f"{output_dir}/basic_metrics.csv"
            metrics_df.to_csv(metrics_path, index=False)
            print(f"✓ Basic metrics saved to {metrics_path}")
        
        # 2. Consistency metrics
        consistency = self.analysis_results.get('consistency', {})
        if consistency:
            # Remove list items for CSV
            consistency_clean = {k: v for k, v in consistency.items() if not isinstance(v, list)}
            consistency_df = pd.DataFrame([consistency_clean])
            consistency_path = f"{output_dir}/consistency_metrics.csv"
            consistency_df.to_csv(consistency_path, index=False)
            print(f"✓ Consistency metrics saved to {consistency_path}")
        
        # 3. Problematic questions
        problematic = self.identify_problematic_questions()
        if not problematic.empty:
            problematic_path = f"{output_dir}/problematic_questions.csv"
            problematic.to_csv(problematic_path, index=False)
            print(f"✓ Problematic questions saved to {problematic_path} ({len(problematic)} questions)")
        
        # 4. Verdict analysis (jika ada)
        if all(col in self.df.columns for col in ['rag_verdict', 'og_verdict']):
            verdict_analysis = []
            
            for verdict in ['BENAR', 'SALAH', 'TIDAK PASTI']:
                rag_count = len(self.df[self.df['rag_verdict'] == verdict])
                og_count = len(self.df[self.df['og_verdict'] == verdict])
                
                verdict_analysis.append({
                    'verdict': verdict,
                    'rag_count': rag_count,
                    'og_count': og_count,
                    'rag_percentage': rag_count / len(self.df) * 100,
                    'og_percentage': og_count / len(self.df) * 100
                })
            
            verdict_df = pd.DataFrame(verdict_analysis)
            verdict_path = f"{output_dir}/verdict_analysis.csv"
            verdict_df.to_csv(verdict_path, index=False)
            print(f"✓ Verdict analysis saved to {verdict_path}")
        
        # 5. Summary report
        summary = {
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'source_file': self.csv_path,
            'total_questions': len(self.df),
            'available_columns': ', '.join(self.df.columns.tolist())
        }
        
        # Add key metrics to summary
        summary.update(basic_metrics)
        summary.update({k: v for k, v in consistency.items() if not isinstance(v, list)})
        
        summary_df = pd.DataFrame([summary])
        summary_path = f"{output_dir}/analysis_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"✓ Analysis summary saved to {summary_path}")
    
    def run_complete_analysis(self, output_dir: str = "analysis_output"):
        """Jalankan analisis lengkap dan generate CSV + PNG output."""
        
        print("🔍 Memulai analisis konsistensi AI...")
        print(f"📁 Output akan disimpan di: {output_dir}")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Run analyses
        print("📊 Menghitung metrik dasar...")
        basic_metrics = self.calculate_basic_metrics()
        
        print("🔄 Menganalisis konsistensi...")
        consistency_metrics = self.analyze_consistency()
        
        # Generate plots
        print("📈 Membuat visualisasi...")
        plot_dir = f"{output_dir}/plots"
        self.generate_simple_plots(plot_dir)
        
        # Save to CSV
        print("💾 Menyimpan hasil ke CSV...")
        self.save_results_to_csv(output_dir)
        
        print(f"\n✅ Analisis selesai!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 Check CSV files dan PNG plots untuk hasil detail")
        
        return {
            'basic_metrics': basic_metrics,
            'consistency_metrics': consistency_metrics,
            'output_directory': output_dir,
            'files_generated': os.listdir(output_dir)
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Consistency Analyzer - Lightweight Version")
    parser.add_argument("csv_file", help="Path to CSV file from RAG vs OG tester")
    parser.add_argument("--output-dir", default="analysis_output", 
                       help="Output directory for analysis results")
    parser.add_argument("--min-score-diff", type=float, default=0.3,
                       help="Minimum score difference to identify problematic questions")
    
    args = parser.parse_args()
    
    try:
        # Run analysis
        analyzer = AIConsistencyAnalyzer(args.csv_file)
        results = analyzer.run_complete_analysis(args.output_dir)
        
        print("\n🎉 Analysis completed successfully!")
        print(f"Generated files: {results['files_generated']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTips:")
        print("- Pastikan file CSV ada dan tidak kosong")
        print("- Periksa format CSV dan header kolom")
        print("- Gunakan file output dari tester RAG vs OG")