#!/usr/bin/env python3
"""
Analisis Kinerja RAG vs OG - Python Implementation
Replicates the functionality of analisis.html for analyzing CSV evaluation data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from typing import Dict, List, Tuple
import warnings
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import re
from collections import Counter, defaultdict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

class RAGEvaluationAnalyzer:
    """
    Analyzer for RAG vs OG evaluation data from CSV files.
    Replicates the functionality from analisis.html JavaScript code.
    """
    
    def __init__(self, eval_directory: str = "eval", output_directory: str = "output"):
        """
        Initialize the analyzer with evaluation directory path.
        
        Args:
            eval_directory (str): Path to directory containing CSV evaluation files
            output_directory (str): Path to directory for saving output files
        """
        self.eval_directory = Path(eval_directory)
        self.output_directory = Path(output_directory)
        self.processed_data = {}
        self.stats_summary = {}
        
        # Create output directory if it doesn't exist
        self.output_directory.mkdir(exist_ok=True)
        
        # Set up matplotlib style for better visualizations
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        print(f"📁 Output directory: {self.output_directory.absolute()}")
    
    def _save_plot(self, filename: str, fig=None):
        """
        Save the current plot as PNG and SVG files.
        
        Args:
            filename (str): Base filename without extension
            fig: matplotlib figure object (optional, uses current figure if None)
        """
        if fig is None:
            fig = plt.gcf()
        
        # Clean filename
        clean_filename = filename.replace(' ', '_').replace('/', '_').replace('\\', '_')
        
        # Save as PNG
        png_path = self.output_directory / f"{clean_filename}.png"
        fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        # Save as SVG
        svg_path = self.output_directory / f"{clean_filename}.svg"
        fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
        
        print(f"  ✓ Saved: {png_path.name} & {svg_path.name}")
        
    def load_csv_files(self) -> Dict[str, pd.DataFrame]:
        """
        Load all CSV files from the evaluation directory.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping filename to DataFrame
        """
        csv_files = glob.glob(str(self.eval_directory / "*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.eval_directory}")
        
        print(f"Found {len(csv_files)} CSV files:")
        
        for csv_file in csv_files:
            filename = Path(csv_file).name
            try:
                # Read CSV with proper handling of quoted strings
                df = pd.read_csv(csv_file, dtype=str)
                
                # Validate required columns
                required_columns = ['pertanyaan', 'rag_benar', 'og_benar']
                if not all(col in df.columns for col in required_columns):
                    print(f"Warning: {filename} missing required columns. Skipping.")
                    continue
                
                # Clean and standardize boolean values
                df['rag_benar'] = df['rag_benar'].str.upper().map({'TRUE': True, 'FALSE': False})
                df['og_benar'] = df['og_benar'].str.upper().map({'TRUE': True, 'FALSE': False})
                
                # Remove rows with invalid boolean values
                df = df.dropna(subset=['rag_benar', 'og_benar'])
                
                self.processed_data[filename] = df
                print(f"  ✓ {filename}: {len(df)} questions")
                
            except Exception as e:
                print(f"  ✗ Error loading {filename}: {e}")
        
        return self.processed_data
    
    def calculate_stats(self, df: pd.DataFrame) -> Dict:
        """
        Calculate statistics for a dataset, replicating JavaScript calculateStats function.
        
        Args:
            df (pd.DataFrame): DataFrame with evaluation data
            
        Returns:
            Dict: Statistics dictionary with counts and percentages
        """
        total = len(df)
        
        if total == 0:
            return {
                'total': 0, 'rag_true': 0, 'og_true': 0, 'both_true': 0,
                'only_rag_true': 0, 'only_og_true': 0, 'both_false': 0,
                'rag_true_percent': 0, 'og_true_percent': 0, 'both_true_percent': 0,
                'only_rag_true_percent': 0, 'only_og_true_percent': 0, 'both_false_percent': 0
            }
        
        # Count different categories
        rag_true = df['rag_benar'].sum()
        og_true = df['og_benar'].sum()
        both_true = ((df['rag_benar'] == True) & (df['og_benar'] == True)).sum()
        only_rag_true = ((df['rag_benar'] == True) & (df['og_benar'] == False)).sum()
        only_og_true = ((df['rag_benar'] == False) & (df['og_benar'] == True)).sum()
        both_false = ((df['rag_benar'] == False) & (df['og_benar'] == False)).sum()
        
        # Calculate percentages
        stats = {
            'total': total,
            'rag_true': rag_true,
            'og_true': og_true,
            'both_true': both_true,
            'only_rag_true': only_rag_true,
            'only_og_true': only_og_true,
            'both_false': both_false,
            'rag_true_percent': round((rag_true / total) * 100, 1),
            'og_true_percent': round((og_true / total) * 100, 1),
            'both_true_percent': round((both_true / total) * 100, 1),
            'only_rag_true_percent': round((only_rag_true / total) * 100, 1),
            'only_og_true_percent': round((only_og_true / total) * 100, 1),
            'both_false_percent': round((both_false / total) * 100, 1)
        }
        
        return stats
    
    def analyze_all_files(self):
        """
        Analyze all loaded CSV files and calculate statistics.
        """
        print("\n" + "="*60)
        print("ANALISIS KINERJA RAG vs OG - HASIL EVALUASI")
        print("="*60)
        
        for filename, df in self.processed_data.items():
            stats = self.calculate_stats(df)
            self.stats_summary[filename] = stats
            
            print(f"\n📊 {filename}")
            print("-" * 50)
            print(f"Total Pertanyaan: {stats['total']}")
            print(f"RAG Benar: {stats['rag_true']} ({stats['rag_true_percent']}%)")
            print(f"OG Benar: {stats['og_true']} ({stats['og_true_percent']}%)")
            print(f"Keduanya Benar: {stats['both_true']} ({stats['both_true_percent']}%)")
            print(f"Hanya RAG Benar: {stats['only_rag_true']} ({stats['only_rag_true_percent']}%)")
            print(f"Hanya OG Benar: {stats['only_og_true']} ({stats['only_og_true_percent']}%)")
            print(f"Keduanya Salah: {stats['both_false']} ({stats['both_false_percent']}%)")
    
    def create_performance_chart(self, filename: str = None):
        """
        Create performance comparison chart (bar chart).
        
        Args:
            filename (str): Specific file to analyze, or None for all files
        """
        if filename and filename in self.stats_summary:
            files_to_plot = {filename: self.stats_summary[filename]}
            title_suffix = f" - {filename}"
        else:
            files_to_plot = self.stats_summary
            title_suffix = " - Semua File"
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        categories = ['RAG Benar', 'OG Benar', 'Keduanya Benar']
        colors = ['#4361ee', '#f72585', '#4cc9f0']
        
        x = np.arange(len(files_to_plot))
        width = 0.25
        
        rag_values = [stats['rag_true'] for stats in files_to_plot.values()]
        og_values = [stats['og_true'] for stats in files_to_plot.values()]
        both_values = [stats['both_true'] for stats in files_to_plot.values()]
        
        ax.bar(x - width, rag_values, width, label='RAG Benar', color=colors[0], alpha=0.8)
        ax.bar(x, og_values, width, label='OG Benar', color=colors[1], alpha=0.8)
        ax.bar(x + width, both_values, width, label='Keduanya Benar', color=colors[2], alpha=0.8)
        
        ax.set_xlabel('File Evaluasi')
        ax.set_ylabel('Jumlah Jawaban Benar')
        ax.set_title(f'📊 Perbandingan Kinerja RAG vs OG{title_suffix}')
        ax.set_xticks(x)
        ax.set_xticklabels([f.replace('.csv', '') for f in files_to_plot.keys()], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        chart_name = f"performance_chart_{filename.replace('.csv', '')}" if filename else "performance_chart_all_files"
        self._save_plot(chart_name)
        
        plt.show()
    
    def create_category_distribution_chart(self, filename: str = None):
        """
        Create category distribution pie chart.
        
        Args:
            filename (str): Specific file to analyze, or None for combined data
        """
        if filename and filename in self.stats_summary:
            stats = self.stats_summary[filename]
            title_suffix = f" - {filename}"
        else:
            # Combine all files
            combined_stats = {
                'both_true': sum(s['both_true'] for s in self.stats_summary.values()),
                'only_rag_true': sum(s['only_rag_true'] for s in self.stats_summary.values()),
                'only_og_true': sum(s['only_og_true'] for s in self.stats_summary.values()),
                'both_false': sum(s['both_false'] for s in self.stats_summary.values())
            }
            stats = combined_stats
            title_suffix = " - Gabungan Semua File"
        
        labels = ['Keduanya Benar', 'Hanya RAG Benar', 'Hanya OG Benar', 'Keduanya Salah']
        sizes = [stats['both_true'], stats['only_rag_true'], stats['only_og_true'], stats['both_false']]
        colors = ['#4cc9f0', '#4895ef', '#f8961e', '#8d99ae']
        
        # Filter out zero values
        non_zero_data = [(label, size, color) for label, size, color in zip(labels, sizes, colors) if size > 0]
        
        if not non_zero_data:
            print("No data to display in pie chart")
            return
        
        labels, sizes, colors = zip(*non_zero_data)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                         startangle=90, explode=[0.05] * len(sizes))
        
        ax.set_title(f'🍩 Distribusi Kategori Jawaban{title_suffix}', fontsize=14, fontweight='bold')
        
        # Beautify the text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        # Save the plot
        chart_name = f"category_distribution_{filename.replace('.csv', '')}" if filename else "category_distribution_all_files"
        self._save_plot(chart_name)
        
        plt.show()
    
    def create_detailed_analysis_table(self, filename: str):
        """
        Create detailed analysis table for a specific file.
        
        Args:
            filename (str): Name of the CSV file to analyze
        """
        if filename not in self.processed_data:
            print(f"File {filename} not found in processed data")
            return
        
        df = self.processed_data[filename].copy()
        
        # Add category column
        def categorize_answer(row):
            rag = row['rag_benar']
            og = row['og_benar']
            
            if rag and og:
                return "Keduanya Benar"
            elif rag and not og:
                return "Hanya RAG Benar"
            elif not rag and og:
                return "Hanya OG Benar"
            else:
                return "Keduanya Salah"
        
        df['kategori'] = df.apply(categorize_answer, axis=1)
        
        print(f"\n📋 ANALISIS DETAIL - {filename}")
        print("="*80)
        
        # Display table with proper formatting
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 60)
        
        for idx, row in df.iterrows():
            print(f"\n{idx+1:2d}. {row['pertanyaan'][:70]}{'...' if len(row['pertanyaan']) > 70 else ''}")
            print(f"    RAG: {'✓' if row['rag_benar'] else '✗'}  |  OG: {'✓' if row['og_benar'] else '✗'}  |  Kategori: {row['kategori']}")
    
    def generate_summary_report(self):
        """
        Generate a comprehensive summary report.
        """
        print("\n" + "="*80)
        print("🏆 KESIMPULAN ANALISIS KINERJA RAG vs OG")
        print("="*80)
        
        if not self.stats_summary:
            print("Tidak ada data untuk dianalisis.")
            return
        
        # Calculate overall statistics
        total_questions = sum(stats['total'] for stats in self.stats_summary.values())
        total_rag_true = sum(stats['rag_true'] for stats in self.stats_summary.values())
        total_og_true = sum(stats['og_true'] for stats in self.stats_summary.values())
        total_both_true = sum(stats['both_true'] for stats in self.stats_summary.values())
        
        overall_rag_percent = round((total_rag_true / total_questions) * 100, 1) if total_questions > 0 else 0
        overall_og_percent = round((total_og_true / total_questions) * 100, 1) if total_questions > 0 else 0
        
        print(f"\n📊 RINGKASAN KESELURUHAN:")
        print(f"   • Total Pertanyaan: {total_questions}")
        print(f"   • Total File Evaluasi: {len(self.stats_summary)}")
        print(f"   • Akurasi RAG: {overall_rag_percent}% ({total_rag_true}/{total_questions})")
        print(f"   • Akurasi OG: {overall_og_percent}% ({total_og_true}/{total_questions})")
        print(f"   • Keduanya Benar: {total_both_true} pertanyaan")
        
        # Determine winner
        if overall_rag_percent > overall_og_percent:
            winner = "RAG"
            margin = overall_rag_percent - overall_og_percent
        elif overall_og_percent > overall_rag_percent:
            winner = "OG"
            margin = overall_og_percent - overall_rag_percent
        else:
            winner = "SERI"
            margin = 0
        
        print(f"\n🏆 HASIL:")
        if winner == "SERI":
            print("   Kinerja RAG dan OG setara!")
        else:
            print(f"   {winner} mengungguli dengan margin {margin}%")
        
        # File-by-file comparison
        print(f"\n📁 PERBANDINGAN PER FILE:")
        for filename, stats in self.stats_summary.items():
            rag_better = stats['rag_true'] > stats['og_true']
            og_better = stats['og_true'] > stats['rag_true']
            
            if rag_better:
                status = f"RAG unggul ({stats['rag_true']} vs {stats['og_true']})"
            elif og_better:
                status = f"OG unggul ({stats['og_true']} vs {stats['rag_true']})"
            else:
                status = f"Seri ({stats['rag_true']} vs {stats['og_true']})"
            
            print(f"   • {filename}: {status}")
    
    # =================== ADVANCED ANALYSIS METHODS ===================
    
    def analyze_consistency_across_files(self):
        """Analyze consistency of performance across different evaluation files."""
        print("\n" + "="*80)
        print("📊 ANALISIS KONSISTENSI ANTAR FILE")
        print("="*80)
        
        if len(self.processed_data) < 2:
            print("Minimal 2 file diperlukan untuk analisis konsistensi.")
            return
        
        file_names = list(self.processed_data.keys())
        rag_scores = [self.stats_summary[f]['rag_true_percent'] for f in file_names]
        og_scores = [self.stats_summary[f]['og_true_percent'] for f in file_names]
        
        print(f"\n📈 KONSISTENSI PERFORMA:")
        print(f"   • Standar Deviasi RAG: {np.std(rag_scores):.2f}%")
        print(f"   • Standar Deviasi OG: {np.std(og_scores):.2f}%")
        print(f"   • Range RAG: {min(rag_scores):.1f}% - {max(rag_scores):.1f}%")
        print(f"   • Range OG: {min(og_scores):.1f}% - {max(og_scores):.1f}%")
    
    def analyze_question_categories(self):
        """Analyze performance by question categories based on keywords."""
        print("\n" + "="*80)
        print("🎯 ANALISIS KATEGORI PERTANYAAN")
        print("="*80)
        
        categories = {
            'Identitas_UKRI': ['ukri', 'universitas kebangsaan', 'singkatan', 'alamat'],
            'Sejarah': ['kapan', 'tahun', 'didirikan', 'berdiri', 'berubah'],
            'Struktur_Organisasi': ['rektor', 'dekan', 'ketua', 'wakil'],
            'Fakultas_Program': ['fakultas', 'program studi', 'fiksi', 'fti', 'ftsp'],
            'Akreditasi': ['akreditasi', 'status', 'lembaga']
        }
        
        category_results = defaultdict(lambda: {'rag_correct': 0, 'og_correct': 0, 'total': 0})
        
        for filename, df in self.processed_data.items():
            for idx, row in df.iterrows():
                question = row['pertanyaan'].lower()
                categorized = False
                
                for category, keywords in categories.items():
                    if any(keyword in question for keyword in keywords):
                        category_results[category]['total'] += 1
                        category_results[category]['rag_correct'] += int(row['rag_benar'])
                        category_results[category]['og_correct'] += int(row['og_benar'])
                        categorized = True
                        break
                
                if not categorized:
                    category_results['Lainnya']['total'] += 1
                    category_results['Lainnya']['rag_correct'] += int(row['rag_benar'])
                    category_results['Lainnya']['og_correct'] += int(row['og_benar'])
        
        print(f"\n📊 PERFORMA PER KATEGORI:")
        category_data = []
        for category, data in category_results.items():
            if data['total'] > 0:
                rag_percent = (data['rag_correct'] / data['total']) * 100
                og_percent = (data['og_correct'] / data['total']) * 100
                
                print(f"\n🏷️  {category.replace('_', ' ')}:")
                print(f"   • Total: {data['total']} pertanyaan")
                print(f"   • RAG: {rag_percent:.1f}% | OG: {og_percent:.1f}%")
                
                category_data.append({
                    'category': category.replace('_', ' '),
                    'rag_percent': rag_percent,
                    'og_percent': og_percent,
                    'total': data['total']
                })
        
        if category_data:
            df_cat = pd.DataFrame(category_data)
            fig, ax = plt.subplots(figsize=(12, 8))
            x = np.arange(len(df_cat))
            width = 0.35
            
            ax.bar(x - width/2, df_cat['rag_percent'], width, label='RAG', color='#4361ee', alpha=0.8)
            ax.bar(x + width/2, df_cat['og_percent'], width, label='OG', color='#f72585', alpha=0.8)
            
            ax.set_xlabel('Kategori Pertanyaan')
            ax.set_ylabel('Persentase Akurasi (%)')
            ax.set_title('🎯 Performa RAG vs OG per Kategori')
            ax.set_xticks(x)
            ax.set_xticklabels(df_cat['category'], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    def analyze_agreement_disagreement(self):
        """Analyze agreement and disagreement patterns between RAG and OG."""
        print("\n" + "="*80)
        print("⚖️ ANALISIS AGREEMENT/DISAGREEMENT")
        print("="*80)
        
        all_rag = []
        all_og = []
        
        for df in self.processed_data.values():
            all_rag.extend(df['rag_benar'].tolist())
            all_og.extend(df['og_benar'].tolist())
        
        kappa = cohen_kappa_score(all_rag, all_og)
        cm = confusion_matrix(all_rag, all_og)
        
        print(f"\n📊 STATISTIK AGREEMENT:")
        print(f"   • Cohen's Kappa: {kappa:.3f}")
        
        if kappa < 0.2:
            agreement_level = "Buruk (Poor)"
        elif kappa < 0.4:
            agreement_level = "Sedang (Fair)"
        elif kappa < 0.6:
            agreement_level = "Baik (Moderate)"
        elif kappa < 0.8:
            agreement_level = "Sangat Baik (Substantial)"
        else:
            agreement_level = "Sempurna (Almost Perfect)"
        
        print(f"   • Level Agreement: {agreement_level}")
        
        total = len(all_rag)
        both_correct = sum(1 for r, o in zip(all_rag, all_og) if r and o)
        both_incorrect = sum(1 for r, o in zip(all_rag, all_og) if not r and not o)
        rag_only = sum(1 for r, o in zip(all_rag, all_og) if r and not o)
        og_only = sum(1 for r, o in zip(all_rag, all_og) if not r and o)
        
        print(f"\n🎯 POLA AGREEMENT:")
        print(f"   • Keduanya Benar: {both_correct} ({both_correct/total*100:.1f}%)")
        print(f"   • Keduanya Salah: {both_incorrect} ({both_incorrect/total*100:.1f}%)")
        print(f"   • Hanya RAG Benar: {rag_only} ({rag_only/total*100:.1f}%)")
        print(f"   • Hanya OG Benar: {og_only} ({og_only/total*100:.1f}%)")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['OG False', 'OG True'],
                   yticklabels=['RAG False', 'RAG True'], ax=ax)
        ax.set_title('⚖️ Confusion Matrix: RAG vs OG Agreement')
        plt.tight_layout()
        
        # Save the plot
        self._save_plot("agreement_disagreement_analysis")
        
        plt.show()
    
    def create_performance_heatmap(self):
        """Create heatmap visualization of performance across files."""
        print("\n" + "="*80)
        print("🔥 HEATMAP PERFORMA")
        print("="*80)
        
        # Prepare data for heatmap
        file_names = list(self.processed_data.keys())
        metrics = ['RAG Accuracy', 'OG Accuracy', 'Both Correct', 'Agreement']
        
        heatmap_data = []
        for filename in file_names:
            stats = self.stats_summary[filename]
            df = self.processed_data[filename]
            
            # Calculate agreement for this file
            agreement = sum(1 for _, row in df.iterrows() 
                          if row['rag_benar'] == row['og_benar']) / len(df) * 100
            
            heatmap_data.append([
                stats['rag_true_percent'],
                stats['og_true_percent'], 
                stats['both_true_percent'],
                agreement
            ])
        
        heatmap_df = pd.DataFrame(heatmap_data, 
                                 index=[f.replace('.csv', '') for f in file_names],
                                 columns=metrics)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Percentage (%)'}, ax=ax)
        ax.set_title('🔥 Heatmap Performa RAG vs OG Across Files')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Evaluation Files')
        plt.tight_layout()
        
        # Save the plot
        self._save_plot("performance_heatmap")
        
        plt.show()
    
    def run_advanced_analysis(self):
        """Run all advanced analysis methods."""
        print("\n" + "="*100)
        print("🚀 MENJALANKAN ANALISIS LANJUTAN")
        print("="*100)
        
        try:
            self.analyze_consistency_across_files()
            self.analyze_question_categories()
            self.analyze_agreement_disagreement()
            self.create_performance_heatmap()
            
            print("\n" + "="*80)
            print("✅ ANALISIS LANJUTAN SELESAI")
            print("="*80)
            
        except Exception as e:
            print(f"❌ Error dalam analisis lanjutan: {e}")
    
    def analyze_per_question_consistency(self):
        """
        Analyze consistency of each question across all test files.
        Shows how many times RAG and OG answered each question correctly.
        """
        print("\n" + "="*80)
        print("🔍 ANALISIS KONSISTENSI PER PERTANYAAN")
        print("="*80)
        
        if not self.processed_data:
            print("Tidak ada data untuk dianalisis.")
            return
        
        # Get the first file to determine number of questions
        first_file = list(self.processed_data.values())[0]
        total_questions = len(first_file)
        total_files = len(self.processed_data)
        
        print(f"📊 Menganalisis {total_questions} pertanyaan dari {total_files} file test...")
        
        # Initialize consistency tracking
        question_consistency = []
        
        for q_idx in range(total_questions):
            rag_correct_count = 0
            og_correct_count = 0
            both_correct_count = 0
            both_wrong_count = 0
            question_text = ""
            
            # Count correct answers across all files for this question
            for file_idx, (filename, df) in enumerate(self.processed_data.items()):
                if q_idx < len(df):
                    row = df.iloc[q_idx]
                    if file_idx == 0:  # Get question text from first file
                        question_text = row['pertanyaan']
                    
                    rag_correct = row['rag_benar']
                    og_correct = row['og_benar']
                    
                    if rag_correct:
                        rag_correct_count += 1
                    if og_correct:
                        og_correct_count += 1
                    
                    # Count agreement patterns
                    if rag_correct and og_correct:
                        both_correct_count += 1
                    elif not rag_correct and not og_correct:
                        both_wrong_count += 1
            
            # Calculate consistency percentages
            rag_consistency = (rag_correct_count / total_files) * 100
            og_consistency = (og_correct_count / total_files) * 100
            both_correct_consistency = (both_correct_count / total_files) * 100
            both_wrong_consistency = (both_wrong_count / total_files) * 100
            
            question_consistency.append({
                'question_num': q_idx + 1,
                'question_text': question_text,
                'rag_correct_count': rag_correct_count,
                'og_correct_count': og_correct_count,
                'both_correct_count': both_correct_count,
                'both_wrong_count': both_wrong_count,
                'rag_consistency': rag_consistency,
                'og_consistency': og_consistency,
                'both_correct_consistency': both_correct_consistency,
                'both_wrong_consistency': both_wrong_consistency,
                'total_files': total_files
            })
        
        # Display detailed results
        print(f"\n📋 HASIL KONSISTENSI PER PERTANYAAN:")
        print("-" * 120)
        print(f"{'No':<3} {'RAG':<8} {'OG':<8} {'Keduanya':<10} {'Keduanya':<10} {'Pertanyaan':<50}")
        print(f"{'':3} {'':8} {'':8} {'Benar':<10} {'Salah':<10} {'':50}")
        print("-" * 120)
        
        for q_data in question_consistency:
            question_short = q_data['question_text'][:45] + "..." if len(q_data['question_text']) > 45 else q_data['question_text']
            rag_display = f"{q_data['rag_correct_count']}/{total_files}"
            og_display = f"{q_data['og_correct_count']}/{total_files}"
            both_correct_display = f"{q_data['both_correct_count']}/{total_files}"
            both_wrong_display = f"{q_data['both_wrong_count']}/{total_files}"
            
            print(f"{q_data['question_num']:<3} {rag_display:<8} {og_display:<8} {both_correct_display:<10} {both_wrong_display:<10} {question_short}")
        
        # Analyze patterns
        self._analyze_consistency_patterns(question_consistency, total_files)
        
        # Create visualizations
        self._create_consistency_visualizations(question_consistency)
        
        return question_consistency
    
    def _analyze_consistency_patterns(self, question_consistency, total_files):
        """
        Analyze patterns in question consistency data.
        """
        print(f"\n🎯 ANALISIS POLA KONSISTENSI:")
        print("-" * 50)
        
        # Perfect consistency (always correct)
        rag_perfect = [q for q in question_consistency if q['rag_correct_count'] == total_files]
        og_perfect = [q for q in question_consistency if q['og_correct_count'] == total_files]
        
        # Never correct
        rag_never = [q for q in question_consistency if q['rag_correct_count'] == 0]
        og_never = [q for q in question_consistency if q['og_correct_count'] == 0]
        
        # Inconsistent (sometimes correct, sometimes not)
        rag_inconsistent = [q for q in question_consistency if 0 < q['rag_correct_count'] < total_files]
        og_inconsistent = [q for q in question_consistency if 0 < q['og_correct_count'] < total_files]
        
        print(f"\n✅ PERTANYAAN SELALU BENAR:")
        print(f"   • RAG: {len(rag_perfect)} pertanyaan ({len(rag_perfect)/len(question_consistency)*100:.1f}%)")
        print(f"   • OG:  {len(og_perfect)} pertanyaan ({len(og_perfect)/len(question_consistency)*100:.1f}%)")
        
        print(f"\n❌ PERTANYAAN SELALU SALAH:")
        print(f"   • RAG: {len(rag_never)} pertanyaan ({len(rag_never)/len(question_consistency)*100:.1f}%)")
        print(f"   • OG:  {len(og_never)} pertanyaan ({len(og_never)/len(question_consistency)*100:.1f}%)")
        
        print(f"\n⚡ PERTANYAAN TIDAK KONSISTEN:")
        print(f"   • RAG: {len(rag_inconsistent)} pertanyaan ({len(rag_inconsistent)/len(question_consistency)*100:.1f}%)")
        print(f"   • OG:  {len(og_inconsistent)} pertanyaan ({len(og_inconsistent)/len(question_consistency)*100:.1f}%)")
        
        # Show most problematic questions
        if rag_inconsistent or og_inconsistent:
            print(f"\n🔥 PERTANYAAN PALING TIDAK KONSISTEN:")
            
            # Sort by inconsistency (closest to 50% success rate)
            rag_most_inconsistent = sorted(rag_inconsistent, 
                                         key=lambda x: abs(x['rag_consistency'] - 50))[:3]
            og_most_inconsistent = sorted(og_inconsistent, 
                                        key=lambda x: abs(x['og_consistency'] - 50))[:3]
            
            if rag_most_inconsistent:
                print(f"\n   RAG - Paling Tidak Konsisten:")
                for q in rag_most_inconsistent:
                    q_short = q['question_text'][:50] + "..." if len(q['question_text']) > 50 else q['question_text']
                    print(f"   • Q{q['question_num']}: {q['rag_correct_count']}/{total_files} - {q_short}")
            
            if og_most_inconsistent:
                print(f"\n   OG - Paling Tidak Konsisten:")
                for q in og_most_inconsistent:
                    q_short = q['question_text'][:50] + "..." if len(q['question_text']) > 50 else q['question_text']
                    print(f"   • Q{q['question_num']}: {q['og_correct_count']}/{total_files} - {q_short}")
        
        # Agreement analysis
        both_perfect = [q for q in question_consistency 
                       if q['rag_correct_count'] == total_files and q['og_correct_count'] == total_files]
        both_never = [q for q in question_consistency 
                     if q['rag_correct_count'] == 0 and q['og_correct_count'] == 0]
        
        print(f"\n🤝 AGREEMENT PATTERNS:")
        print(f"   • Keduanya Selalu Benar: {len(both_perfect)} pertanyaan")
        print(f"   • Keduanya Selalu Salah: {len(both_never)} pertanyaan")
        
        if both_perfect:
            print(f"\n   ✅ Pertanyaan Termudah (Keduanya Selalu Benar):")
            for q in both_perfect[:5]:  # Show first 5
                q_short = q['question_text'][:60] + "..." if len(q['question_text']) > 60 else q['question_text']
                print(f"   • Q{q['question_num']}: {q_short}")
        
        if both_never:
            print(f"\n   ❌ Pertanyaan Tersulit (Keduanya Selalu Salah):")
            for q in both_never[:5]:  # Show first 5
                q_short = q['question_text'][:60] + "..." if len(q['question_text']) > 60 else q['question_text']
                print(f"   • Q{q['question_num']}: {q_short}")
    
    def _create_consistency_visualizations(self, question_consistency):
        """
        Create visualizations for question consistency analysis.
        """
        # Extract data for plotting
        question_nums = [q['question_num'] for q in question_consistency]
        rag_counts = [q['rag_correct_count'] for q in question_consistency]
        og_counts = [q['og_correct_count'] for q in question_consistency]
        both_correct_counts = [q['both_correct_count'] for q in question_consistency]
        both_wrong_counts = [q['both_wrong_count'] for q in question_consistency]
        total_files = question_consistency[0]['total_files']
        
        # Create comprehensive visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Line chart showing consistency across questions
        ax1.plot(question_nums, rag_counts, 'o-', label='RAG Benar', color='#4361ee', alpha=0.7, linewidth=2)
        ax1.plot(question_nums, og_counts, 's-', label='OG Benar', color='#f72585', alpha=0.7, linewidth=2)
        ax1.plot(question_nums, both_correct_counts, '^-', label='Keduanya Benar', color='#4cc9f0', alpha=0.7, linewidth=2)
        ax1.plot(question_nums, both_wrong_counts, 'v-', label='Keduanya Salah', color='#8d99ae', alpha=0.7, linewidth=2)
        ax1.axhline(y=total_files, color='green', linestyle='--', alpha=0.5, label='Perfect (All Tests)')
        ax1.axhline(y=total_files/2, color='orange', linestyle='--', alpha=0.5, label='50% Consistency')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Never Correct')
        ax1.set_xlabel('Nomor Pertanyaan')
        ax1.set_ylabel(f'Jumlah (dari {total_files} test)')
        ax1.set_title('🔍 Konsistensi Jawaban per Pertanyaan')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.5, total_files + 0.5)
        
        # Plot 2: Histogram of consistency levels
        ax2.hist(rag_counts, bins=range(total_files+2), alpha=0.6, label='RAG Benar', color='#4361ee', edgecolor='black')
        ax2.hist(og_counts, bins=range(total_files+2), alpha=0.6, label='OG Benar', color='#f72585', edgecolor='black')
        ax2.hist(both_correct_counts, bins=range(total_files+2), alpha=0.6, label='Keduanya Benar', color='#4cc9f0', edgecolor='black')
        ax2.hist(both_wrong_counts, bins=range(total_files+2), alpha=0.6, label='Keduanya Salah', color='#8d99ae', edgecolor='black')
        ax2.set_xlabel(f'Jumlah Test (dari {total_files})')
        ax2.set_ylabel('Jumlah Pertanyaan')
        ax2.set_title('📊 Distribusi Tingkat Konsistensi')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Scatter plot RAG vs OG consistency
        ax3.scatter(rag_counts, og_counts, alpha=0.6, s=50, color='purple')
        ax3.plot([0, total_files], [0, total_files], 'r--', alpha=0.5, label='Perfect Agreement')
        ax3.set_xlabel(f'RAG Correct Count (dari {total_files})')
        ax3.set_ylabel(f'OG Correct Count (dari {total_files})')
        ax3.set_title('⚖️ RAG vs OG Consistency Correlation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(-0.5, total_files + 0.5)
        ax3.set_ylim(-0.5, total_files + 0.5)
        
        # Plot 4: Difference between RAG and OG
        differences = [r - o for r, o in zip(rag_counts, og_counts)]
        colors = ['green' if d > 0 else 'red' if d < 0 else 'gray' for d in differences]
        ax4.bar(question_nums, differences, color=colors, alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.8)
        ax4.set_xlabel('Nomor Pertanyaan')
        ax4.set_ylabel('RAG - OG (Selisih Jumlah Benar)')
        ax4.set_title('📈 Selisih Performa RAG vs OG per Pertanyaan')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_plot("per_question_consistency_analysis")
        plt.show()
        
        # Create summary statistics
        print(f"\n📈 STATISTIK KONSISTENSI:")
        print(f"   • Rata-rata RAG benar: {np.mean(rag_counts):.1f}/{total_files} ({np.mean(rag_counts)/total_files*100:.1f}%)")
        print(f"   • Rata-rata OG benar: {np.mean(og_counts):.1f}/{total_files} ({np.mean(og_counts)/total_files*100:.1f}%)")
        print(f"   • Standar deviasi RAG: {np.std(rag_counts):.2f}")
        print(f"   • Standar deviasi OG: {np.std(og_counts):.2f}")
        print(f"   • Korelasi RAG-OG: {np.corrcoef(rag_counts, og_counts)[0,1]:.3f}")
    
    def create_per_batch_question_matrix(self):
        """
        Create matrix visualization showing correctness for each question across all batches.
        Shows which questions were correct/wrong in each test batch for both RAG and OG.
        """
        print("\n" + "="*80)
        print("📋 MATRIX BENAR/SALAH PER BATCH PER PERTANYAAN")
        print("="*80)
        
        if not self.processed_data:
            print("Tidak ada data untuk dianalisis.")
            return
        
        # Get data dimensions
        file_names = list(self.processed_data.keys())
        first_file = list(self.processed_data.values())[0]
        total_questions = len(first_file)
        total_files = len(self.processed_data)
        
        print(f"📊 Membuat matrix {total_questions} pertanyaan x {total_files} batch test...")
        
        # Create matrices for RAG and OG
        rag_matrix = np.zeros((total_questions, total_files), dtype=int)
        og_matrix = np.zeros((total_questions, total_files), dtype=int)
        agreement_matrix = np.zeros((total_questions, total_files), dtype=int)
        
        question_texts = []
        
        # Fill matrices
        for file_idx, (filename, df) in enumerate(self.processed_data.items()):
            for q_idx in range(min(total_questions, len(df))):
                row = df.iloc[q_idx]
                
                if file_idx == 0:  # Get question text from first file
                    question_texts.append(row['pertanyaan'])
                
                rag_correct = int(row['rag_benar'])
                og_correct = int(row['og_benar'])
                
                rag_matrix[q_idx, file_idx] = rag_correct
                og_matrix[q_idx, file_idx] = og_correct
                
                # Agreement matrix: 2=both correct, 1=one correct, 0=both wrong
                if rag_correct and og_correct:
                    agreement_matrix[q_idx, file_idx] = 2  # Both correct
                elif rag_correct or og_correct:
                    agreement_matrix[q_idx, file_idx] = 1  # One correct
                else:
                    agreement_matrix[q_idx, file_idx] = 0  # Both wrong
        
        # Create visualizations
        self._create_batch_question_heatmaps(rag_matrix, og_matrix, agreement_matrix, 
                                           file_names, question_texts)
        
        # Print summary statistics
        self._print_batch_question_summary(rag_matrix, og_matrix, agreement_matrix, 
                                          file_names, question_texts)
        
        return rag_matrix, og_matrix, agreement_matrix
    
    def _create_batch_question_heatmaps(self, rag_matrix, og_matrix, agreement_matrix, 
                                       file_names, question_texts):
        """
        Create a single combined heatmap visualization for RAG and OG correctness.
        """
        # Create combined matrix: RAG and OG side by side
        total_files = len(file_names)
        combined_matrix = np.zeros((len(question_texts), total_files * 2))
        
        # Fill combined matrix: first half RAG, second half OG
        combined_matrix[:, :total_files] = rag_matrix
        combined_matrix[:, total_files:] = og_matrix
        
        # Create single figure
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Prepare labels
        file_labels = [f.replace('.csv', '').replace('RagEvaluation', 'Test') for f in file_names]
        combined_labels = [f'RAG-{label}' for label in file_labels] + [f'OG-{label}' for label in file_labels]
        question_labels = [f"Q{i+1}" for i in range(len(question_texts))]
        
        # Create heatmap
        im = ax.imshow(combined_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_title('📋 Matrix Benar/Salah per Batch - RAG vs OG', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Batch Test (RAG | OG)', fontsize=12)
        ax.set_ylabel('Pertanyaan', fontsize=12)
        
        # Set x-axis ticks and labels
        ax.set_xticks(range(len(combined_labels)))
        ax.set_xticklabels(combined_labels, rotation=45, ha='right')
        
        # Set y-axis ticks (show every 5th question for readability)
        y_ticks = range(0, len(question_labels), 5)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([question_labels[i] for i in y_ticks])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Benar (1) / Salah (0)', fontsize=12)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Salah ✗', 'Benar ✓'])
        
        # Add vertical line to separate RAG and OG
        ax.axvline(x=total_files-0.5, color='black', linewidth=2, alpha=0.8)
        
        # Add text labels for RAG and OG sections
        ax.text(total_files/2 - 0.5, -2, 'RAG', ha='center', va='top', 
               fontsize=14, fontweight='bold', color='#4361ee')
        ax.text(total_files + total_files/2 - 0.5, -2, 'OG', ha='center', va='top', 
               fontsize=14, fontweight='bold', color='#f72585')
        
        # Add grid for better readability
        ax.set_xticks(np.arange(-0.5, len(combined_labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(question_labels), 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        self._save_plot("per_batch_question_matrix")
        
        plt.show()
    
    def _print_batch_question_summary(self, rag_matrix, og_matrix, agreement_matrix, 
                                    file_names, question_texts):
        """
        Print summary statistics for the batch-question analysis.
        """
        print(f"\n📊 RINGKASAN MATRIX ANALYSIS:")
        print("-" * 60)
        
        # Per-batch statistics
        print(f"\n📁 PERFORMA PER BATCH:")
        for i, filename in enumerate(file_names):
            rag_correct = np.sum(rag_matrix[:, i])
            og_correct = np.sum(og_matrix[:, i])
            both_correct = np.sum(agreement_matrix[:, i] == 2)
            both_wrong = np.sum(agreement_matrix[:, i] == 0)
            
            batch_name = filename.replace('.csv', '').replace('RagEvaluation', 'Test')
            print(f"   • {batch_name}: RAG {rag_correct}/{len(question_texts)} | "
                  f"OG {og_correct}/{len(question_texts)} | "
                  f"Both✓ {both_correct} | Both✗ {both_wrong}")
        
        # Most/least consistent questions
        question_consistency = []
        for q_idx in range(len(question_texts)):
            rag_correct = np.sum(rag_matrix[q_idx, :])
            og_correct = np.sum(og_matrix[q_idx, :])
            both_correct = np.sum(agreement_matrix[q_idx, :] == 2)
            both_wrong = np.sum(agreement_matrix[q_idx, :] == 0)
            
            consistency_score = both_correct + both_wrong  # High when they agree
            
            question_consistency.append({
                'q_num': q_idx + 1,
                'text': question_texts[q_idx],
                'rag_correct': rag_correct,
                'og_correct': og_correct,
                'both_correct': both_correct,
                'both_wrong': both_wrong,
                'consistency_score': consistency_score
            })
        
        # Sort by consistency
        sorted_by_consistency = sorted(question_consistency, key=lambda x: x['consistency_score'])
        
        print(f"\n🔥 PERTANYAAN PALING TIDAK KONSISTEN (Sering Disagreement):")
        for q in sorted_by_consistency[:5]:
            q_short = q['text'][:50] + "..." if len(q['text']) > 50 else q['text']
            print(f"   • Q{q['q_num']}: RAG {q['rag_correct']}/{len(file_names)} | "
                  f"OG {q['og_correct']}/{len(file_names)} - {q_short}")
        
        print(f"\n✅ PERTANYAAN PALING KONSISTEN (Sering Agreement):")
        for q in sorted_by_consistency[-5:]:
            q_short = q['text'][:50] + "..." if len(q['text']) > 50 else q['text']
            agreement_type = "Keduanya Benar" if q['both_correct'] > q['both_wrong'] else "Keduanya Salah"
            print(f"   • Q{q['q_num']}: {agreement_type} - {q_short}")
        
        # Overall statistics
        total_cells = rag_matrix.size
        rag_total_correct = np.sum(rag_matrix)
        og_total_correct = np.sum(og_matrix)
        both_correct_total = np.sum(agreement_matrix == 2)
        both_wrong_total = np.sum(agreement_matrix == 0)
        
        print(f"\n📊 STATISTIK KESELURUHAN:")
        print(f"   • Total Cells: {total_cells} (50 pertanyaan x 8 batch)")
        print(f"   • RAG Correct: {rag_total_correct}/{total_cells} ({rag_total_correct/total_cells*100:.1f}%)")
        print(f"   • OG Correct: {og_total_correct}/{total_cells} ({og_total_correct/total_cells*100:.1f}%)")
        print(f"   • Both Correct: {both_correct_total}/{total_cells} ({both_correct_total/total_cells*100:.1f}%)")
        print(f"   • Both Wrong: {both_wrong_total}/{total_cells} ({both_wrong_total/total_cells*100:.1f}%)")
        print(f"   • Agreement Rate: {(both_correct_total + both_wrong_total)/total_cells*100:.1f}%")
    
    def run_complete_analysis(self):
        """
        Run the complete analysis pipeline.
        """
        print("🚀 Memulai Analisis Kinerja RAG vs OG...")
        
        # Load data
        self.load_csv_files()
        
        if not self.processed_data:
            print("❌ Tidak ada data yang berhasil dimuat. Analisis dihentikan.")
            return
        
        # Analyze all files
        self.analyze_all_files()
        
        # Generate visualizations
        print("\n📈 Membuat visualisasi...")
        self.create_performance_chart()
        self.create_category_distribution_chart()
        
        # Generate summary
        self.generate_summary_report()
        
        # Interactive options
        print("\n" + "="*60)
        print("💡 OPSI ANALISIS LANJUTAN:")
        print("   1. Lihat detail file tertentu")
        print("   2. Buat grafik untuk file tertentu")
        print("   3. 🚀 Jalankan Analisis Lanjutan (Konsistensi, Kategori, Agreement, Heatmap)")
        print("   4. 🔍 Analisis Konsistensi Per Pertanyaan (Detail Batch Test)")
        print("   5. 📋 Matrix Benar/Salah per Batch per Pertanyaan (50x8 Matrix)")
        print("   6. 📊 Analisis Konsistensi Antar File")
        print("   7. 🎯 Analisis Kategori Pertanyaan")
        print("   8. ⚖️ Analisis Agreement/Disagreement")
        print("   9. 🔥 Heatmap Performa")
        print("   10. Keluar")
        
        while True:
            try:
                choice = input("\nPilih opsi (1-10): ").strip()
                
                if choice == "1":
                    print("\nFile yang tersedia:")
                    for i, filename in enumerate(self.processed_data.keys(), 1):
                        print(f"   {i}. {filename}")
                    
                    file_choice = input("Pilih nomor file: ").strip()
                    try:
                        file_idx = int(file_choice) - 1
                        filename = list(self.processed_data.keys())[file_idx]
                        self.create_detailed_analysis_table(filename)
                    except (ValueError, IndexError):
                        print("Pilihan tidak valid.")
                
                elif choice == "2":
                    print("\nFile yang tersedia:")
                    for i, filename in enumerate(self.processed_data.keys(), 1):
                        print(f"   {i}. {filename}")
                    
                    file_choice = input("Pilih nomor file: ").strip()
                    try:
                        file_idx = int(file_choice) - 1
                        filename = list(self.processed_data.keys())[file_idx]
                        self.create_performance_chart(filename)
                        self.create_category_distribution_chart(filename)
                    except (ValueError, IndexError):
                        print("Pilihan tidak valid.")
                
                elif choice == "3":
                    print("🚀 Menjalankan semua analisis lanjutan...")
                    self.run_advanced_analysis()
                
                elif choice == "4":
                    print("🔍 Menganalisis konsistensi per pertanyaan...")
                    self.analyze_per_question_consistency()
                
                elif choice == "5":
                    print("📋 Membuat matrix benar/salah per batch per pertanyaan...")
                    self.create_per_batch_question_matrix()
                
                elif choice == "6":
                    self.analyze_consistency_across_files()
                
                elif choice == "7":
                    self.analyze_question_categories()
                
                elif choice == "8":
                    self.analyze_agreement_disagreement()
                
                elif choice == "9":
                    self.create_performance_heatmap()
                
                elif choice == "10":
                    print("👋 Analisis selesai. Terima kasih!")
                    break
                
                else:
                    print("Pilihan tidak valid. Silakan pilih 1-10.")
                    
            except KeyboardInterrupt:
                print("\n👋 Analisis dihentikan. Terima kasih!")
                break


def main():
    """
    Main function to run the RAG evaluation analysis.
    """
    # Initialize analyzer with eval directory
    analyzer = RAGEvaluationAnalyzer("eval")
    
    try:
        # Run complete analysis
        analyzer.run_complete_analysis()
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Pastikan folder 'eval' ada dan berisi file CSV dengan format yang benar.")
        
    except Exception as e:
        print(f"❌ Terjadi kesalahan: {e}")
        print("💡 Periksa format file CSV dan coba lagi.")


if __name__ == "__main__":
    main()
