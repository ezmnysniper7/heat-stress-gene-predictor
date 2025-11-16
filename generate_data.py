"""
Generate simulated gene expression data for heat stress response study
This script creates a realistic dataset mimicking RNA-seq differential expression results
"""

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of genes to simulate
n_genes = 500

# Generate gene IDs
gene_ids = [f"GENE_{i:05d}" for i in range(1, n_genes + 1)]

# Simulate log2FoldChange values
# Most genes have small changes, some have large changes (stress-responsive)
log2fc = np.random.normal(0, 1.5, n_genes)
# Add some truly differentially expressed genes
n_responsive = 100
responsive_indices = np.random.choice(n_genes, n_responsive, replace=False)
log2fc[responsive_indices[:50]] = np.random.uniform(2, 5, 50)  # Upregulated
log2fc[responsive_indices[50:]] = np.random.uniform(-5, -2, 50)  # Downregulated

# Simulate p-values
# Responsive genes tend to have lower p-values
p_values = np.random.uniform(0.001, 0.5, n_genes)
for idx in responsive_indices:
    p_values[idx] = np.random.uniform(0.0001, 0.01)  # Significant p-values

# Simulate GC content (typical range 40-60% for plant genes)
gc_content = np.random.normal(50, 8, n_genes)
gc_content = np.clip(gc_content, 30, 70)  # Clip to realistic range

# Simulate gene length (base pairs)
gene_length = np.random.lognormal(7, 1.5, n_genes).astype(int)
gene_length = np.clip(gene_length, 200, 15000)

# Simulate expression level (baseMean - average normalized counts)
base_mean = np.random.lognormal(5, 2, n_genes)

# Create binary label: responsive if |log2FC| > 1 AND p_value < 0.05
responsive = ((np.abs(log2fc) > 1) & (p_values < 0.05)).astype(int)

# Create DataFrame
data = pd.DataFrame({
    'gene_id': gene_ids,
    'log2FoldChange': log2fc,
    'p_value': p_values,
    'GC_content': gc_content,
    'gene_length': gene_length,
    'baseMean': base_mean,
    'responsive': responsive
})

# Add adjusted p-value (FDR correction simulation)
from scipy.stats import false_discovery_control
data['padj'] = false_discovery_control(data['p_value'])

# Add gene annotations (simulate some known heat shock proteins)
gene_names = []
for i, gene_id in enumerate(gene_ids):
    if responsive[i] == 1 and np.random.random() > 0.7:
        # Some responsive genes are labeled as heat shock proteins
        hsp_type = np.random.choice(['HSP70', 'HSP90', 'HSF', 'DREB', 'LEA'])
        gene_names.append(f"{hsp_type}_{np.random.randint(1, 20)}")
    else:
        gene_names.append(f"Gene_{i+1}")

data['gene_name'] = gene_names

# Reorder columns
data = data[['gene_id', 'gene_name', 'log2FoldChange', 'p_value', 'padj',
             'GC_content', 'gene_length', 'baseMean', 'responsive']]

# Save to CSV
data.to_csv('heat_stress_data.csv', index=False)

# Print summary statistics
print("=" * 60)
print("HEAT STRESS GENE EXPRESSION DATA - SUMMARY")
print("=" * 60)
print(f"\nTotal genes: {n_genes}")
print(f"Responsive genes: {responsive.sum()} ({responsive.sum()/n_genes*100:.1f}%)")
print(f"Non-responsive genes: {(1-responsive).sum()}")
print(f"\nlog2FoldChange range: [{log2fc.min():.2f}, {log2fc.max():.2f}]")
print(f"p-value range: [{p_values.min():.6f}, {p_values.max():.6f}]")
print(f"GC content range: [{gc_content.min():.1f}%, {gc_content.max():.1f}%]")
print("\nFirst few rows:")
print(data.head(10))
print("\nClass distribution:")
print(data['responsive'].value_counts())
print("\n✓ Data saved to 'heat_stress_data.csv'")
print("=" * 60)
