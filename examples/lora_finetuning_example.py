#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example script demonstrating LoRA fine-tuning with Time-MoE

This script shows how to use LoRA (Low-Rank Adaptation) from the transformers/peft
library to perform parameter-efficient fine-tuning of Time-MoE models.

Usage:
    python examples/lora_finetuning_example.py

Requirements:
    - peft>=0.10.0
    - All other Time-MoE requirements
"""

import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

# ============================================================================
# Example 1: Basic LoRA Setup for Inference
# ============================================================================

def example_lora_inference():
    """
    Example showing how to load a base Time-MoE model and apply LoRA configuration
    """
    print("=" * 80)
    print("Example 1: Basic LoRA Setup")
    print("=" * 80)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        'Maple728/TimeMoE-50M',
        device_map="cpu",
        trust_remote_code=True,
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # Time-MoE is a causal language model
        r=8,                            # LoRA rank (lower = fewer parameters)
        lora_alpha=16,                  # LoRA scaling factor
        lora_dropout=0.05,              # Dropout probability
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention layers
        bias="none",                    # Don't adapt bias parameters
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    print("\nLoRA model created successfully!")
    print("Note: Only LoRA parameters will be trained during fine-tuning.")
    
    return model


# ============================================================================
# Example 2: LoRA Configuration Options
# ============================================================================

def example_lora_configurations():
    """
    Example showing different LoRA configurations for various use cases
    """
    print("\n" + "=" * 80)
    print("Example 2: Different LoRA Configurations")
    print("=" * 80)
    
    # Configuration 1: Minimal parameters (fastest, lowest memory)
    minimal_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    print("\n1. Minimal LoRA Configuration:")
    print(f"   - Rank: {minimal_config.r}")
    print(f"   - Alpha: {minimal_config.lora_alpha}")
    print(f"   - Target modules: {minimal_config.target_modules}")
    print(f"   - Use case: Quick experiments, limited compute")
    
    # Configuration 2: Balanced (recommended for most cases)
    balanced_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    print("\n2. Balanced LoRA Configuration:")
    print(f"   - Rank: {balanced_config.r}")
    print(f"   - Alpha: {balanced_config.lora_alpha}")
    print(f"   - Target modules: {balanced_config.target_modules}")
    print(f"   - Use case: Most fine-tuning tasks")
    
    # Configuration 3: High capacity (best performance)
    high_capacity_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    print("\n3. High Capacity LoRA Configuration:")
    print(f"   - Rank: {high_capacity_config.r}")
    print(f"   - Alpha: {high_capacity_config.lora_alpha}")
    print(f"   - Target modules: {high_capacity_config.target_modules}")
    print(f"   - Use case: Complex tasks, sufficient compute")


# ============================================================================
# Example 3: Time-MoE Specific Target Modules
# ============================================================================

def example_target_modules():
    """
    Example showing different target module combinations for Time-MoE
    """
    print("\n" + "=" * 80)
    print("Example 3: Time-MoE Target Modules")
    print("=" * 80)
    
    print("\nAvailable target modules in Time-MoE:")
    print("  - Attention layers:")
    print("    - q_proj, k_proj, v_proj, o_proj (query, key, value, output projections)")
    print("\n  - MoE expert layers:")
    print("    - gate_proj, up_proj, down_proj (expert network layers)")
    print("\n  - Other layers:")
    print("    - gate (MoE router/gate)")
    print("    - shared_expert (shared expert in MoE)")
    
    print("\nRecommended combinations:")
    print("  1. Attention only: ['q_proj', 'k_proj', 'v_proj', 'o_proj']")
    print("  2. Attention + FFN: ['q_proj', 'v_proj', 'gate_proj', 'down_proj']")
    print("  3. Full coverage: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']")


# ============================================================================
# Example 4: Training with LoRA (using CLI)
# ============================================================================

def example_training_commands():
    """
    Example showing how to use LoRA with the Time-MoE training script
    """
    print("\n" + "=" * 80)
    print("Example 4: Training Commands with LoRA")
    print("=" * 80)
    
    print("\n1. Basic LoRA fine-tuning:")
    print("   python main.py -d <data_path> --use_lora")
    
    print("\n2. Custom LoRA configuration:")
    print("   python main.py -d <data_path> --use_lora --lora_r 16 --lora_alpha 32")
    
    print("\n3. LoRA with custom target modules:")
    print("   python main.py -d <data_path> --use_lora --lora_target_modules q_proj,k_proj,v_proj,o_proj")
    
    print("\n4. LoRA with other training options:")
    print("   python main.py -d <data_path> \\")
    print("       --use_lora \\")
    print("       --lora_r 8 \\")
    print("       --lora_alpha 16 \\")
    print("       --learning_rate 1e-4 \\")
    print("       --num_train_epochs 3 \\")
    print("       --global_batch_size 64")
    
    print("\n5. Multi-GPU training with LoRA:")
    print("   python torch_dist_run.py main.py -d <data_path> --use_lora")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Time-MoE LoRA Fine-tuning Examples")
    print("=" * 80)
    
    try:
        # Run non-model examples (model examples require downloading the model)
        example_lora_configurations()
        example_target_modules()
        example_training_commands()
        
        # Uncomment to run actual model examples (requires downloading the model)
        # These are provided as reference but not run by default to avoid
        # downloading large model files during example execution
        # example_lora_inference()
        # example_inference_test()
        
        print("\n" + "=" * 80)
        print("All examples completed!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Install peft: pip install peft")
        print("  2. Prepare your dataset in jsonl format")
        print("  3. Run training with LoRA: python main.py -d <data_path> --use_lora")
        print("\nFor hands-on model examples, uncomment the example_lora_inference()")
        print("and example_inference_test() calls in this script.")
        print("\n")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure you have installed all requirements: pip install -r requirements.txt")
