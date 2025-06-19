from tokenizer import GPTTokenizer
import argparse
import torch

def main():
    parser = argparse.ArgumentParser(description='Encode/decode text using a trained tokenizer')
    parser.add_argument('--tokenizer', type=str, required=True, help='Tokenizer file')
    parser.add_argument('--text', type=str, help='Text to encode')
    parser.add_argument('--file', type=str, help='File containing text to encode')
    parser.add_argument('--decode', action='store_true', help='Decode instead of encode')
    parser.add_argument('--tokens', type=str, help='Tokens to decode (comma separated)')
    
    args = parser.parse_args()
    
    # Load tokenizer
    tokenizer = GPTTokenizer.load(args.tokenizer)
    
    if args.decode:
        # Decode tokens
        if args.tokens:
            token_ids = [int(t) for t in args.tokens.split(',')]
        elif args.text:
            token_ids = [int(t) for t in args.text.split(',')]
        else:
            raise ValueError("No tokens provided for decoding")
        
        decoded = tokenizer.decode(torch.tensor(token_ids))
        print("Decoded text:", decoded)
    else:
        # Encode text
        if args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        elif args.text:
            text = args.text
        else:
            raise ValueError("No text provided for encoding")
        
        encoded = tokenizer.encode(text)
        print("Encoded tokens:", encoded.tolist())

if __name__ == "__main__":
    main()