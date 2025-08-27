#!/usr/bin/env python3
"""N-gram text generation script - Task #7 Implementation - SELF-CONTAINED"""
import argparse
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description='Generate text using n-gram model')
    parser.add_argument('--prompt', type=str, default="To be", 
                       help='Text prompt for generation')
    parser.add_argument('--max_tokens', type=int, default=50, 
                       help='Maximum tokens to generate')
    parser.add_argument('--model', type=str, default='models/best_ngram.pkl', 
                       help='Path to saved model')
    parser.add_argument('--mode', type=str, default='sample', choices=['argmax', 'sample'],
                       help='Generation mode: argmax or sample')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    
    args = parser.parse_args()
    
    try:
        # Import here to avoid dependency issues
        from models.ngram import NGramModel
        
        # Load model
        model = NGramModel().load(args.model)
        
        if not hasattr(model.engine, 'generate'):
            print("Error: Model engine does not support generation")
            sys.exit(1)
        
        # Generate text (argmax + optional sampling, EOS handling)
        generated = model.engine.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            mode=args.mode,
            temperature=args.temperature
        )
        
        print(f"Prompt: {args.prompt}")
        print(f"Generated: {' '.join(generated)}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the model file exists and was trained properly.")
        sys.exit(1)

if __name__ == "__main__":
    main()
