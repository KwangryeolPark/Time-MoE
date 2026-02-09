#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
Example script demonstrating how to extract latent representations from Time-MoE model.

This script shows three different ways to obtain latent representations:
1. Using the convenient encode() method
2. Using the forward() method with output_hidden_states=True
3. Using the generate() method with output_hidden_states=True
"""

import torch
from transformers import AutoModelForCausalLM


def example_1_using_encode():
    """Example 1: Using the encode() method (recommended)"""
    print("=" * 80)
    print("Example 1: Extracting latent representations using encode() method")
    print("=" * 80)
    
    # Load model (using small model for quick testing)
    model = AutoModelForCausalLM.from_pretrained(
        'Maple728/TimeMoE-50M',
        device_map="cpu",
        trust_remote_code=True,
    )
    
    # Prepare input
    context_length = 12
    batch_size = 2
    seqs = torch.randn(batch_size, context_length)
    
    # Normalize sequences
    mean = seqs.mean(dim=-1, keepdim=True)
    std = seqs.std(dim=-1, keepdim=True)
    normed_seqs = (seqs - mean) / std
    
    # Extract latent representation using encode method
    print(f"\nInput shape: {normed_seqs.shape}")
    outputs = model.encode(normed_seqs)
    
    # Get the final hidden state (latent representation)
    latent_repr = outputs.last_hidden_state
    print(f"Latent representation shape: {latent_repr.shape}")
    print(f"  - Batch size: {latent_repr.shape[0]}")
    print(f"  - Sequence length: {latent_repr.shape[1]}")
    print(f"  - Hidden dimension: {latent_repr.shape[2]}")
    
    # Access all layer hidden states
    if outputs.hidden_states is not None:
        print(f"\nNumber of layer outputs (including input embeddings): {len(outputs.hidden_states)}")
        print(f"Each layer output shape: {outputs.hidden_states[0].shape}")
        
        # You can access specific layer outputs
        first_layer_output = outputs.hidden_states[0]  # After input embedding
        middle_layer_output = outputs.hidden_states[len(outputs.hidden_states) // 2]
        final_layer_output = outputs.hidden_states[-1]  # Same as last_hidden_state
        
        print(f"\nFirst layer output shape: {first_layer_output.shape}")
        print(f"Middle layer output shape: {middle_layer_output.shape}")
        print(f"Final layer output shape: {final_layer_output.shape}")
    
    return latent_repr


def example_2_using_forward():
    """Example 2: Using the forward() method with output_hidden_states=True"""
    print("\n" + "=" * 80)
    print("Example 2: Extracting latent representations using forward() method")
    print("=" * 80)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        'Maple728/TimeMoE-50M',
        device_map="cpu",
        trust_remote_code=True,
    )
    
    # Prepare input
    context_length = 12
    batch_size = 2
    seqs = torch.randn(batch_size, context_length)
    
    # Normalize sequences
    mean = seqs.mean(dim=-1, keepdim=True)
    std = seqs.std(dim=-1, keepdim=True)
    normed_seqs = (seqs - mean) / std
    
    # Forward pass with output_hidden_states=True
    print(f"\nInput shape: {normed_seqs.shape}")
    outputs = model.forward(
        input_ids=normed_seqs,
        output_hidden_states=True,
        return_dict=True,
    )
    
    # Get the final hidden state (latent representation)
    latent_repr = outputs.hidden_states[-1] if outputs.hidden_states else None
    
    if latent_repr is not None:
        print(f"Latent representation shape: {latent_repr.shape}")
        print(f"Number of layer outputs: {len(outputs.hidden_states)}")
    else:
        print("Hidden states not available. Make sure output_hidden_states=True")
    
    return latent_repr


def example_3_using_generate():
    """Example 3: Using the generate() method with output_hidden_states=True"""
    print("\n" + "=" * 80)
    print("Example 3: Extracting latent representations during generation")
    print("=" * 80)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        'Maple728/TimeMoE-50M',
        device_map="cpu",
        trust_remote_code=True,
    )
    
    # Prepare input
    context_length = 12
    batch_size = 2
    seqs = torch.randn(batch_size, context_length)
    
    # Normalize sequences
    mean = seqs.mean(dim=-1, keepdim=True)
    std = seqs.std(dim=-1, keepdim=True)
    normed_seqs = (seqs - mean) / std
    
    # Generate with output_hidden_states=True
    print(f"\nInput shape: {normed_seqs.shape}")
    
    prediction_length = 6
    outputs = model.generate(
        normed_seqs,
        max_new_tokens=prediction_length,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    
    # With generate, hidden_states contains states from each generation step
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
        print(f"Generated sequence shape: {outputs.sequences.shape}")
        print(f"Number of generation steps: {len(outputs.hidden_states)}")
        
        # Each element in hidden_states is a tuple of layer outputs for that generation step
        if outputs.hidden_states[0]:
            print(f"Number of layers per step: {len(outputs.hidden_states[0])}")
            print(f"First step, first layer shape: {outputs.hidden_states[0][0].shape}")
    
    return outputs


def example_4_use_cases():
    """Example 4: Common use cases for latent representations"""
    print("\n" + "=" * 80)
    print("Example 4: Common use cases for latent representations")
    print("=" * 80)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        'Maple728/TimeMoE-50M',
        device_map="cpu",
        trust_remote_code=True,
    )
    
    # Prepare multiple time series
    num_series = 3
    context_length = 24
    time_series_data = [
        torch.randn(1, context_length) for _ in range(num_series)
    ]
    
    print(f"\nExtracting representations from {num_series} time series")
    
    # Extract representations for each series
    representations = []
    for i, series in enumerate(time_series_data):
        # Normalize
        mean = series.mean(dim=-1, keepdim=True)
        std = series.std(dim=-1, keepdim=True)
        normed_series = (series - mean) / std
        
        # Encode
        outputs = model.encode(normed_series)
        latent = outputs.last_hidden_state
        
        # Use mean pooling to get a fixed-size representation
        pooled_repr = latent.mean(dim=1)  # [1, hidden_size]
        representations.append(pooled_repr)
        
        print(f"Series {i+1}: Input shape {series.shape} -> Latent shape {latent.shape} -> Pooled shape {pooled_repr.shape}")
    
    # Stack all representations
    all_repr = torch.cat(representations, dim=0)  # [num_series, hidden_size]
    print(f"\nAll representations stacked: {all_repr.shape}")
    
    # Use cases:
    print("\nPossible use cases for these representations:")
    print("1. Similarity computation: torch.nn.functional.cosine_similarity(repr1, repr2)")
    print("2. Clustering: Use representations as features for K-means, DBSCAN, etc.")
    print("3. Classification: Train a classifier on top of the representations")
    print("4. Anomaly detection: Compare new representations with normal ones")
    print("5. Dimensionality reduction: Apply PCA, t-SNE, or UMAP for visualization")
    print("6. Transfer learning: Use as features for downstream tasks")
    
    return all_repr


if __name__ == "__main__":
    print("\nTime-MoE Latent Representation Extraction Examples")
    print("=" * 80)
    print("\nThese examples demonstrate how to extract latent representations")
    print("(hidden states) from the Time-MoE model for various purposes.")
    print()
    
    # Run examples
    latent_1 = example_1_using_encode()
    latent_2 = example_2_using_forward()
    latent_3 = example_3_using_generate()
    all_repr = example_4_use_cases()
    
    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
