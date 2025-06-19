from tokenizer import GPTTokenizer
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
    parser.add_argument('--input', type=str, required=True, help='Input text file')
    parser.add_argument('--output', type=str, required=True, help='Output tokenizer file')
    parser.add_argument('--vocab-size', type=int, default=300, help='Vocabulary size')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Read input text
    with open(args.input, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Initialize and train tokenizer
    tokenizer = GPTTokenizer()
    
    # Add special tokens (customize as needed)
    tokenizer.add_special_token("<|endoftext|>")
    tokenizer.add_special_token("<|padding|>")
    
    # Train the tokenizer
    tokenizer.train(text, vocab_size=args.vocab_size, verbose=args.verbose)
    
    # Save the tokenizer
    tokenizer.save(args.output)
    print(f"Tokenizer trained and saved to {args.output}")

if __name__ == "__main__":
    main()